import unittest

import harness.analyze as A
from harness.analyze import analyze
from harness.metrics import holm_correct
from tests.test_analyze import chain, loaded

_SYNTH_PANEL = {f"p{i}" for i in range(12)}


def setUpModule():
    global _saved_panels
    _saved_panels = dict(A.REGISTERED_PANELS)
    A.REGISTERED_PANELS["gsm8k_base"] = set(_SYNTH_PANEL)
    A.REGISTERED_PANELS["letters"] = set(_SYNTH_PANEL)


def tearDownModule():
    A.REGISTERED_PANELS.clear()
    A.REGISTERED_PANELS.update(_saved_panels)


_TEST_MODEL = {"haiku": "claude-haiku-4-5-20251001",
               "sonnet": "claude-sonnet-5"}


def conf_chain(arm, cons, k, pid, r, ans, **kw):
    c = chain(arm, cons, k, pid, r, ans, campaign="SW-JUL26-R4CLEAN", **kw)
    c["_class"] = "clean"
    c["invocation_id"] = "inv1"
    c["config_fp"] = "fp1"
    c["model_ids_observed"] = [_TEST_MODEL.get(arm, f"model-{arm}")]
    order = ["degenerate", "fixed", "finalonly"] if k == 7 else         ["degenerate", "fixed"]
    c["unit_order"] = order
    c["order_position"] = order.index(cons) if cons in order else 0
    return c


def full_clean_arm(arm):
    recs = []
    for pid in [f"p{i}" for i in range(12)]:
        for r in range(5):
            for cons in ("degenerate", "fixed", "finalonly"):
                # degenerate scatters across wrong answers (D collapses);
                # fixed/finalonly stay correct and deterministic
                ans = "10" if cons != "degenerate" else str(20 + r)
                recs.append(conf_chain(arm, cons, 7, pid, r, ans))
            for cons in ("degenerate", "fixed"):
                recs.append(conf_chain(arm, cons, 1, pid, r, "10"))
    return recs


def meta_pass(arm):
    return {"record_type": "meta", "campaign": "SW-JUL26-R4CLEAN",
            "arm": f"{arm}-r4clean", "_group": arm,
            "invocation_id": "inv1", "config_fp": "fp1",
            "git_tree_clean": True,
            "scaffold_status": "PASS", "scaffold_probe": {"gate": "PASS"}}


class TestA9Gates(unittest.TestCase):
    def test_full_clean_wave_succeeds(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        data = {"latest": recs, "all": recs + [meta_pass("haiku"),
                                               meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertTrue(fs["complete"] and fs["success"])

    def test_missing_run_blocks_pc_presence(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        # remove one degenerate k7 run for one haiku problem
        recs = [r for r in recs
                if not (r["_group"] == "haiku" and r["construction"] ==
                        "degenerate" and r["k"] == 7
                        and r["problem_id"] == "p0" and r["run_idx"] == 4)]
        data = {"latest": recs, "all": recs + [meta_pass("haiku"),
                                               meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])
        self.assertIn("haiku", fs["missing_arms"] + fs.get("partial_arms", []))

    def test_missing_k1_fails_gate(self):
        recs = [r for r in full_clean_arm("haiku") + full_clean_arm("sonnet")
                if not (r["_group"] == "haiku" and r["k"] == 1)]
        data = {"latest": recs, "all": recs + [meta_pass("haiku"),
                                               meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])

    def test_missing_scaffold_gate_fails(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        data = {"latest": recs, "all": recs + [meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])
        self.assertIn("haiku", fs.get("scaffold_gate_missing", []))

    def test_reduced_scaffold_relabels(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        m = meta_pass("haiku"); m["scaffold_status"] = "REDUCED_SCAFFOLD"
        data = {"latest": recs, "all": recs + [m, meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])
        self.assertEqual(fs.get("relabel"), "reduced-scaffold")

    def test_abort_marks_incomplete(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        ab = {"record_type": "meta_abort", "campaign": "SW-JUL26-R4CLEAN",
              "arm": "haiku-r4clean", "_group": "haiku", "reason": "x"}
        data = {"latest": recs, "all": recs + [meta_pass("haiku"),
                                               meta_pass("sonnet"), ab]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["complete"] or fs["success"])

    def test_meta_only_wave_reports_incomplete(self):
        data = {"latest": [chain("x", "fixed", 7, "p0", 0, "10")],
                "all": [meta_pass("haiku")]}
        fs = analyze(data)["family_status"].get("clean")
        self.assertIsNotNone(fs)
        self.assertFalse(fs["complete"] or fs["success"])

    def test_holm_rejects_overfull_family(self):
        with self.assertRaises(ValueError):
            holm_correct({"a": 0.01, "b": 0.02, "c": 0.03}, m_total=2)


if __name__ == "__main__":
    unittest.main()


class TestProvenanceValueGate(unittest.TestCase):
    def test_fingerprint_mismatch_fails(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        m1 = meta_pass("haiku"); m1["config_fp"] = "DIFFERENT"
        data = {"latest": recs, "all": recs + [m1, meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])
        self.assertIn("haiku", fs.get("provenance_gaps", []))

    def test_dirty_tree_meta_fails(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        m1 = meta_pass("haiku"); m1["git_tree_clean"] = False
        data = {"latest": recs, "all": recs + [m1, meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])


class TestPanelAndConfigGates(unittest.TestCase):
    def test_wrong_panel_fails(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        for r in recs:
            if r["_group"] == "haiku":
                r["problem_id"] = r["problem_id"].replace("p", "q")
        data = {"latest": recs, "all": recs + [meta_pass("haiku"),
                                               meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])

    def test_mixed_config_fails(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        half = [r for r in recs if r["_group"] == "haiku"][:60]
        for r in half:
            r["config_fp"] = "fp2"
            r["invocation_id"] = "inv2"
        m2 = meta_pass("haiku")
        m2["invocation_id"] = "inv2"; m2["config_fp"] = "fp2"
        data = {"latest": recs, "all": recs + [meta_pass("haiku"), m2,
                                               meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])
        self.assertIn("haiku", fs.get("mixed_config_arms", []))


class TestIdentityGate(unittest.TestCase):
    def test_wrong_model_id_fails(self):
        recs = full_clean_arm("haiku") + full_clean_arm("sonnet")
        for r in recs:
            if r["_group"] == "haiku":
                r["model_ids_observed"] = ["claude-haiku-9-9"]
        data = {"latest": recs, "all": recs + [meta_pass("haiku"),
                                               meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["success"])
        self.assertTrue(fs.get("identity_failures"))

    def test_rogue_arm_voids_family(self):
        recs = (full_clean_arm("haiku") + full_clean_arm("sonnet")
                + full_clean_arm("opus5"))
        data = {"latest": recs, "all": recs + [meta_pass("haiku"),
                                               meta_pass("sonnet")]}
        fs = analyze(data)["family_status"]["clean"]
        self.assertFalse(fs["complete"] or fs["success"])
        self.assertIn("opus5", fs.get("rogue_arms", []))
