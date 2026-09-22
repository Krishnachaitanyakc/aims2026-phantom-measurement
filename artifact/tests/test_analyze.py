import unittest

from harness.analyze import (analyze, cell_stats, equivalence_decision,
                             paired_gap, wilson_interval)


def chain(arm, cons, k, pid, run_idx, answer, gold="10", error=None,
          refusal=False, campaign="SW-JUL26", deviations=0):
    return {
        "record_type": "chain", "arm": arm, "arm_group": arm,
        "campaign": campaign, "_group": arm,
        "_class": "main" if campaign in ("SW-JUL26", "SW-JUL26-R2EXT")
        else "ctl",
        "construction": cons, "k": k,
        "problem_id": pid, "run_idx": run_idx, "answer": answer, "gold": gold,
        "correct": (error is None) and answer == gold,
        "error": error, "any_step_refusal": refusal, "final_refusal": refusal,
        "parse_path": "hash" if answer else "none",
        "model_ids_observed": [f"model-{arm}"], "deviations": deviations,
        "steps": [{}] * k,
    }


def loaded(recs):
    return {"latest": recs, "all": recs}


def build_synthetic():
    chains = []
    problems = [f"p{i}" for i in range(5)]
    for k in (1, 7):
        for pid in problems:
            for r in range(5):
                chains.append(chain("armA", "fixed", k, pid, r, "10"))
                if k == 1:
                    chains.append(chain("armA", "degenerate", k, pid, r, "10"))
                else:
                    chains.append(
                        chain("armA", "degenerate", k, pid, r, str(20 + r)))
    return chains


class TestCellStats(unittest.TestCase):
    def test_equivalence_boundary_roundoff_cannot_pass(self):
        self.assertIsNone(equivalence_decision(-0.09999999999999998, -0.0167, 0.1))
        self.assertIsNone(equivalence_decision(0.0167, 0.09999999999999998, 0.1))
        self.assertTrue(equivalence_decision(-0.09, 0.09, 0.1))
        self.assertFalse(equivalence_decision(-0.11, 0.09, 0.1))

    def test_perfect_cell(self):
        recs = [chain("a", "fixed", 7, "p1", r, "10") for r in range(5)]
        cs = cell_stats(recs)
        self.assertEqual(cs["D_mean"], 1.0)
        self.assertEqual(cs["D_strict_mean"], 1.0)
        self.assertEqual(cs["acc_mean"], 1.0)
        self.assertEqual(cs["scoreable_frac"], 1.0)
        self.assertEqual(cs["acc_strict_mean"], 1.0)
        self.assertIn("D_ci_wilson_problem", cs)
        lo, hi = cs["D_ci_wilson_problem"]
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)

    def test_exclusion_below_min_valid(self):
        recs = [chain("a", "fixed", 7, "p1", r, "10",
                      error="X" if r < 2 else None) for r in range(5)]
        cs = cell_stats(recs)
        self.assertEqual(cs["n_problems_used"], 0)
        self.assertEqual(cs["excluded_problems"], ["p1"])

    def test_strict_departs_from_primary_on_refusals(self):
        # all runs converge on "99" but flagged refusal -> D high, D_strict low
        recs = [chain("a", "degenerate", 7, "p1", r, "99", refusal=True)
                for r in range(5)]
        cs = cell_stats(recs)
        self.assertEqual(cs["D_mean"], 1.0)
        self.assertEqual(cs["D_strict_mean"], 0.2)
        self.assertEqual(cs["scoreable_frac"], 0.0)

    def test_deviation_sensitivity(self):
        recs = [chain("a", "fixed", 7, "p1", r, "10",
                      deviations=1 if r == 0 else 0) for r in range(5)]
        cs = cell_stats(recs)
        self.assertEqual(cs["n_deviation_chains"], 1)
        self.assertEqual(cs["sensitivity_excl_deviations"]["D_mean"], 1.0)


class TestPairedAndAnalyze(unittest.TestCase):
    def test_paired_gap_direction_and_2stage(self):
        fixed = cell_stats([chain("a", "fixed", 7, f"p{i}", r, "10")
                            for i in range(4) for r in range(5)])
        degen = cell_stats([chain("a", "degenerate", 7, f"p{i}", r, str(r))
                            for i in range(4) for r in range(5)])
        gap = paired_gap(fixed, degen, seed=1, two_stage=True)
        self.assertEqual(gap["n"], 4)
        self.assertAlmostEqual(gap["D_gap_mean"], 0.8)
        self.assertGreater(gap["D_gap_ci"][0], 0)
        self.assertIn("D_gap_ci_2stage", gap)
        self.assertGreater(gap["D_gap_ci_2stage"][0], 0)

    def test_full_analyze_synthetic(self):
        summary = analyze(loaded(build_synthetic()))
        prim = summary["primary"]["armA"]
        self.assertEqual(prim["k"], 7)
        self.assertAlmostEqual(prim["D_gap_mean"], 0.8)
        gap_k1 = summary["arms"]["armA"]["main"]["gaps"][
            "fixed_minus_degenerate_k1"]
        self.assertEqual(gap_k1["D_gap_mean"], 0.0)
        self.assertTrue(summary["model_id_check"]["armA"]["uniform_among_observed"])
        self.assertEqual(summary["model_id_check"]["armA"]["n_chains_without_observed_id"], 0)
        acct = summary["accounting"]["armA"]
        self.assertEqual(acct["accepted_chains"], len(build_synthetic()))

    def test_ctl_class_separation_and_finalonly_gap(self):
        recs = []
        for pid in ("p1", "p2", "p3"):
            for r in range(5):
                recs.append(chain("a", "fixed", 7, pid, r, "10",
                                  campaign="SW-JUL26-R2CTL"))
                recs.append(chain("a", "finalonly", 7, pid, r, "10",
                                  campaign="SW-JUL26-R2CTL"))
                recs.append(chain("a", "degenerate", 7, pid, r, str(r),
                                  campaign="SW-JUL26-R2CTL"))
                recs.append(chain("a", "padded2", 7, pid, r, str(r + 10),
                                  campaign="SW-JUL26-R2CTL"))
        summary = analyze(loaded(recs))
        ctl = summary["arms"]["a"]["ctl"]["gaps"]
        self.assertAlmostEqual(
            ctl["fixed_minus_finalonly_k7"]["D_gap_mean"], 0.0)
        self.assertAlmostEqual(
            ctl["finalonly_minus_degenerate_k7"]["D_gap_mean"], 0.8)
        self.assertNotIn("main", summary["arms"]["a"])
        self.assertNotIn("a", summary["primary"])  # ctl never sets primary

    def test_main_pools_disjoint_campaign_problems(self):
        recs = []
        for pid in ("p1", "p2"):
            for r in range(5):
                recs.append(chain("a", "fixed", 7, pid, r, "10"))
                recs.append(chain("a", "degenerate", 7, pid, r, str(r)))
        for pid in ("q1", "q2"):
            for r in range(5):
                recs.append(chain("a", "fixed", 7, pid, r, "10",
                                  campaign="SW-JUL26-R2EXT"))
                recs.append(chain("a", "degenerate", 7, pid, r, str(r),
                                  campaign="SW-JUL26-R2EXT"))
        summary = analyze(loaded(recs))
        self.assertEqual(summary["primary"]["a"]["n"], 4)


class TestRebound(unittest.TestCase):
    def test_rebound_estimand(self):
        recs = []
        for pid in ("p1", "p2", "p3"):
            for r in range(5):
                # k5 scattered, k7 concentrated on one wrong answer
                recs.append(chain("a", "degenerate", 5, pid, r, str(r)))
                recs.append(chain("a", "degenerate", 7, pid, r, "99"))
                recs.append(chain("a", "fixed", 5, pid, r, "10"))
                recs.append(chain("a", "fixed", 7, pid, r, "10"))
                recs.append(chain("a", "fixed", 1, pid, r, "10"))
                recs.append(chain("a", "degenerate", 1, pid, r, "10"))
        summary = analyze(loaded(recs))
        reb = summary["arms"]["a"]["main"]["rebound_deg_k7_minus_k5"]
        self.assertAlmostEqual(reb["D_gap_mean"], 0.8)


class TestWilson(unittest.TestCase):
    def test_wilson_bounds(self):
        lo, hi = wilson_interval(60, 60)
        self.assertGreater(lo, 0.9)
        self.assertEqual(round(hi, 4), 1.0)
        lo2, hi2 = wilson_interval(30, 60)
        self.assertLess(lo2, 0.5)
        self.assertGreater(hi2, 0.5)


class TestFailureModeConcentration(unittest.TestCase):
    def test_high_D_wrong_modal(self):
        recs = [chain("a", "degenerate", 7, "p1", r, "99") for r in range(5)]
        recs += [chain("a", "degenerate", 7, "p2", r, "10") for r in range(5)]
        recs_fixed = [chain("a", "fixed", 7, p, r, "10")
                      for p in ("p1", "p2") for r in range(5)]
        recs_k1 = [chain("a", c, 1, p, r, "10") for c in ("fixed", "degenerate")
                   for p in ("p1", "p2") for r in range(5)]
        summary = analyze(loaded(recs + recs_fixed + recs_k1))
        fm = summary["arms"]["a"]["main"]["failure_mode"]["degenerate"]
        self.assertEqual(fm["n_highD"], 2)
        self.assertEqual(fm["n_highD_modal_wrong"], 1)


if __name__ == "__main__":
    unittest.main()
