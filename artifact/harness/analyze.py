"""Offline analysis retained from the July campaign for numerical continuity.

The AIMS distribution includes only SW-JUL26-R3 GSM8K records. Its entry
point validates that scope before calling this module. Earlier/later class
handlers remain covered by synthetic tests; no missing wave is imputed.

Campaign classes (EXPERIMENT_PLAN.md + Addendum R2):
  main — SW-JUL26 (base problems) pooled with SW-JUL26-R2EXT (disjoint
         addendum problems): primary n=24 endpoints.
  ctl  — SW-JUL26-R2CTL: four constructions in-window at k=7 (n=12);
         fresh-solve (finalonly) and token-matched length (padded2)
         contrasts, plus a second-window replication of fixed−degenerate.
Cells are never merged across classes; dedup keys include the campaign.

Per-cell parser-robustness decomposition (pre-registered R2 endpoint 5):
  scoreable = final answer parsed via '####' AND no final-step refusal;
  D        = max-vote share over all parsed answers (primary, as historical);
  D_strict = max-vote share with non-scoreable chains as distinct
             no-answers (they never agree);
  accuracy = correctness under the permissive numeric parser;
  acc_strict additionally requires a scoreable final answer.
Uncertainty: problem-level BCa (primary), two-stage bootstrap
(problems, then runs within problem) as sensitivity. The boundary Wilson
interval uses the number of problems, not the number of calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean

from .metrics import (bca_ci, determinism_max_vote, holm_correct,
                      paired_permutation_p, tara)
from .parsing import answers_match

ARTIFACT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ARTIFACT_DIR / "runs"
RESULTS_DIR = ARTIFACT_DIR / "results"

MIN_VALID = 4
CONCENTRATION_D = 0.6
BOOT_B = 10_000
BOOT_B_2STAGE = 4_000

MAIN_CAMPAIGNS = {"SW-JUL26", "SW-JUL26-R2EXT"}   # exploratory (relabeled)
CTL_CAMPAIGNS = {"SW-JUL26-R2CTL"}                # exploratory controls
CONF_CAMPAIGNS = {"SW-JUL26-R3"}                  # confirmatory (frozen)
LETTERS_CAMPAIGNS = {"SW-JUL26-R3LET"}            # confirmatory, 2nd task
CLEAN_CAMPAIGNS = {"SW-JUL26-R4CLEAN"}            # scaffold-minimized CLI
R5_CAMPAIGNS = {"SW-JUL26-R5"}                    # remote expanded-model wave
R5LET_CAMPAIGNS = {"SW-JUL26-R5LET"}              # remote letters
# Pre-registered Holm families per class (Amendment A6): fixed a priori;
# a missing arm marks the family incomplete — never silently shrunk.
# Analysis regime per class (Amendment A9): a8 classes use the
# pairwise-complete estimand for presence/Holm/success; a6 classes keep
# their registered permissive primary and are never retroactively re-judged.
REGIME = {"conf": "a6", "letters": "a6", "clean": "a8", "r5": "a8",
          "r5letters": "a8"}


def _load_registered_panels() -> dict:
    """Registered problem-ID panels from the pinned data files (A11)."""
    import json as _json
    panels = {}
    base = ARTIFACT_DIR / "data" / "subset_seed20260728.json"
    if base.exists():
        panels["gsm8k_base"] = set(_json.loads(base.read_text())["ids"])
    let = ARTIFACT_DIR / "data" / "letters_seed20260730.json"
    if let.exists():
        panels["letters"] = {i["id"] for i in
                             _json.loads(let.read_text())["items"]}
    return panels


REGISTERED_PANELS = _load_registered_panels()

# Expected pinned model per arm group (A12). PENDING entries can never
# validate, so un-pinned waves cannot reach success. Effort applies to
# codex groups only.
EXPECTED_MODELS = {
    "haiku": ("claude-haiku-4-5-20251001", None),
    "sonnet": ("claude-sonnet-5", None),
    "codex-gpt56": ("gpt-5.6-sol", "medium"),
    "sonnet46": ("PENDING-PIN-sonnet-4-6", None),
    "opus5": ("PENDING-PIN-opus-5", None),
    "gpt55": ("PENDING-PIN-gpt-5.5", "medium"),
    "gpt54": ("PENDING-PIN-gpt-5.4", "medium"),
    "gpt54mini": ("PENDING-PIN-gpt-5.4-mini", "medium"),
    "gpt52": ("PENDING-PIN-gpt-5.2", "medium"),
}
# Registered full grids (cells that must be full-panel for completeness).
GRIDS = {
    "conf": ["degenerate_k1", "fixed_k1", "degenerate_k5", "fixed_k5",
             "degenerate_k7", "fixed_k7", "finalonly_k7"],
    "letters": ["degenerate_k1", "fixed_k1", "degenerate_k7", "fixed_k7"],
    "clean": ["degenerate_k1", "fixed_k1", "degenerate_k7", "fixed_k7",
              "finalonly_k7"],
    "r5": ["degenerate_k1", "fixed_k1", "degenerate_k7", "fixed_k7",
           "finalonly_k7"],
    "r5letters": ["degenerate_k1", "fixed_k1", "degenerate_k7",
                  "fixed_k7"],
}
FAMILIES = {
    "conf": ["haiku", "codex-gpt56", "sonnet"],
    "letters": ["haiku", "codex-gpt56"],
    "clean": ["haiku", "sonnet"],
    "r5": ["sonnet46", "opus5", "gpt55", "gpt54", "gpt54mini", "gpt52"],
    "r5letters": ["sonnet46", "gpt55"],
}


def _seed_for(*key) -> int:
    h = hashlib.sha256("|".join(str(k) for k in key).encode()).hexdigest()
    return 20260728 + int(h[:8], 16) % 100_000


def equivalence_decision(lo: float, hi: float, margin: float) -> bool | None:
    """Boundary contact is inconclusive, including floating-point roundoff."""
    if (math.isclose(lo, -margin, rel_tol=0, abs_tol=1e-12)
            or math.isclose(hi, margin, rel_tol=0, abs_tol=1e-12)):
        return None
    return -margin < lo and hi < margin


def load_chains(runs_dir: Path, include_smoke: bool = False) -> dict:
    """Latest chain record per logical key; returns {'latest':…, 'all':…}."""
    latest: dict[tuple, dict] = {}
    all_records: list[dict] = []
    for path in sorted(runs_dir.glob("*.jsonl")):
        if path.name.startswith("smoke_") and not include_smoke:
            continue
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") in ("meta", "meta_end", "meta_abort"):
                rec["_group"] = rec.get("arm_group") or rec.get("arm", "")
                all_records.append(rec)
                continue
            if rec.get("record_type") != "chain":
                continue
            group = rec.get("arm_group") or rec["arm"]
            rec["_group"] = group
            campaign = rec.get("campaign", "SW-JUL26")
            rec["_class"] = ("main" if campaign in MAIN_CAMPAIGNS
                             else "ctl" if campaign in CTL_CAMPAIGNS
                             else "conf" if campaign in CONF_CAMPAIGNS
                             else "letters" if campaign in LETTERS_CAMPAIGNS
                             else "clean" if campaign in CLEAN_CAMPAIGNS
                             else "r5" if campaign in R5_CAMPAIGNS
                             else "r5letters" if campaign in R5LET_CAMPAIGNS
                             else "other")
            all_records.append(rec)
            latest[(group, campaign, rec["construction"], rec["k"],
                    rec["problem_id"], rec["run_idx"])] = rec
    return {"latest": list(latest.values()), "all": all_records}


def _scoreable(rec: dict) -> bool:
    return rec["parse_path"] == "hash" and not rec.get("final_refusal")


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cell_stats(chains: list[dict]) -> dict:
    by_problem: dict[str, list[dict]] = {}
    for c in chains:
        by_problem.setdefault(c["problem_id"], []).append(c)
    per_problem: dict[str, dict] = {}
    excluded = []
    for pid, recs in sorted(by_problem.items()):
        valid = sorted((r for r in recs if r["error"] is None),
                       key=lambda r: r.get("run_idx", 0))
        if len(valid) < MIN_VALID:
            excluded.append(pid)
            continue
        # answers are run_idx-ordered so index-aligned resampling across two
        # cells of a pair is genuinely paired (run i with run i)
        answers = [r["answer"] for r in valid]
        strict_answers = [r["answer"] if _scoreable(r) else "" for r in valid]
        modal = max(answers, key=lambda a: sum(
            1 for b in answers if answers_match(a, b)))
        per_problem[pid] = {
            "D": determinism_max_vote(answers),
            "D_strict": determinism_max_vote(strict_answers),
            "acc": mean(1.0 if r["correct"] else 0.0 for r in valid),
            "acc_strict": mean(1.0 if (r["correct"] and _scoreable(r))
                               else 0.0 for r in valid),
            "answers_by_run": {r.get("run_idx", i): r["answer"]
                               for i, r in enumerate(valid)},
            "correct_by_run": {r.get("run_idx", i):
                               (1.0 if r["correct"] else 0.0)
                               for i, r in enumerate(valid)},
            "tara": tara(answers),
            "answers": answers,
            "strict_answers": strict_answers,
            "modal_answer": modal,
            "modal_correct": answers_match(modal, valid[0]["gold"]),
            "n_valid": len(valid),
        }
    # All-problems (incl. excluded) errors-as-failures store (A9)
    per_problem_all = {}
    for pid, recs in sorted(by_problem.items()):
        per_problem_all[pid] = {
            "D_errfail": determinism_max_vote(
                [r["answer"] if r["error"] is None else f"__ERR{i}__"
                 for i, r in enumerate(recs)]),
            "acc_errfail": mean(
                1.0 if (r["error"] is None and r["correct"]) else 0.0
                for r in recs),
            "n_records": len(recs),
        }
    d_vals = [v["D"] for v in per_problem.values()]
    ds_vals = [v["D_strict"] for v in per_problem.values()]
    acc_vals = [v["acc"] for v in per_problem.values()]
    n_chains = len(chains)
    valid_chains = [c for c in chains if c["error"] is None]
    seed = _seed_for("cell", chains[0]["_group"], chains[0]["_class"],
                     chains[0]["construction"], chains[0]["k"]) if chains else 0
    dev_chains = [c for c in valid_chains if c.get("deviations", 0) > 0]
    out = {
        "n_chains": n_chains,
        "n_errors": n_chains - len(valid_chains),
        "n_problems_used": len(per_problem),
        "excluded_problems": excluded,
        "D_mean": round(mean(d_vals), 4) if d_vals else None,
        "D_ci": [round(x, 4) for x in bca_ci(d_vals, BOOT_B, seed=seed)]
        if d_vals else None,
        "D_strict_mean": round(mean(ds_vals), 4) if ds_vals else None,
        "acc_mean": round(mean(acc_vals), 4) if acc_vals else None,
        "acc_ci": [round(x, 4) for x in bca_ci(acc_vals, BOOT_B, seed=seed + 1)]
        if acc_vals else None,
        "tara_mean": round(mean(v["tara"] for v in per_problem.values()), 4)
        if per_problem else None,
        "scoreable_frac": round(mean(
            1.0 if _scoreable(c) else 0.0 for c in valid_chains), 4)
        if valid_chains else None,
        "refusal_rate_any_step": round(mean(
            1.0 if c["any_step_refusal"] else 0.0 for c in valid_chains), 4)
        if valid_chains else None,
        "parse_paths": {p: sum(1 for c in valid_chains
                               if c["parse_path"] == p)
                        for p in ("hash", "fallback", "none")},
        "n_deviation_chains": len(dev_chains),
        "per_problem_all": per_problem_all,
        "per_problem": per_problem,
    }
    # Boundary-degenerate CI: problem-level Wilson on the count of unanimous
    # problems (the problem is the sampling unit; pooling runs would treat
    # correlated draws as independent).
    if d_vals and min(d_vals) == 1.0:
        n_prob = len(per_problem)
        out["D_ci_wilson_problem"] = [round(x, 4) for x in
                                      wilson_interval(n_prob, n_prob)]
    # Accuracy is NOT parser-independent: strict accuracy counts only
    # scoreable-correct chains (permissive accuracy remains the historical
    # instrument's estimand and is reported alongside).
    acc_strict_vals = [mean(1.0 if (r["correct"] and _scoreable(r)) else 0.0
                            for r in recs if r["error"] is None)
                       for pid, recs in sorted(by_problem.items())
                       if pid in per_problem]
    out["acc_strict_mean"] = (round(mean(acc_strict_vals), 4)
                              if acc_strict_vals else None)
    # Fallback-value decomposition: what does the fallback parser extract?
    fb_answers = [c["answer"] for c in valid_chains
                  if c["parse_path"] == "fallback" and c["answer"]]
    if fb_answers:
        from collections import Counter
        top, top_n = Counter(fb_answers).most_common(1)[0]
        k_val = chains[0]["k"]
        step_idx_n = sum(1 for a in fb_answers
                         if a in (str(k_val), str(k_val - 1)))
        out["fallback_mix"] = {
            "n_fallback": len(fb_answers),
            "top_value": top, "top_n": top_n,
            "n_equal_final_step_index": step_idx_n,
        }
    # Deviation sensitivity (pre-registered endpoint 7).
    if dev_chains:
        keep = [c for c in valid_chains if c.get("deviations", 0) == 0]
        by_p: dict[str, list[str]] = {}
        corr: dict[str, list[float]] = {}
        for c in keep:
            by_p.setdefault(c["problem_id"], []).append(c["answer"])
            corr.setdefault(c["problem_id"], []).append(
                1.0 if c["correct"] else 0.0)
        ds = [determinism_max_vote(a) for a in by_p.values() if len(a) >= 2]
        accs = [mean(v) for v in corr.values()]
        out["sensitivity_excl_deviations"] = {
            "D_mean": round(mean(ds), 4) if ds else None,
            "acc_mean": round(mean(accs), 4) if accs else None,
        }
    return out


def paired_gap(cell_a: dict, cell_b: dict, seed: int,
               two_stage: bool = False) -> dict:
    common = sorted(set(cell_a["per_problem"]) & set(cell_b["per_problem"]))
    if not common:
        return {"n": 0}
    ga = cell_a["per_problem"]
    gb = cell_b["per_problem"]
    gaps_d = [ga[p]["D"] - gb[p]["D"] for p in common]
    gaps_ds = [ga[p]["D_strict"] - gb[p]["D_strict"] for p in common]
    gaps_acc = [ga[p]["acc"] - gb[p]["acc"] for p in common]
    gaps_accs = [ga[p]["acc_strict"] - gb[p]["acc_strict"] for p in common]
    complete = [p for p in common
                if ga[p]["n_valid"] == 5 and gb[p]["n_valid"] == 5]
    # Pairwise-complete estimand (Amendment A8): D over the runs shared by
    # BOTH cells; primary for frozen classes (requires all 5 shared).
    pc = [p for p in common
          if len(set(ga[p]["answers_by_run"])
                 & set(gb[p]["answers_by_run"])) == 5]
    def _pc_pair(p):
        shared = sorted(set(ga[p]["answers_by_run"])
                        & set(gb[p]["answers_by_run"]))
        da = determinism_max_vote([ga[p]["answers_by_run"][i]
                                   for i in shared])
        db = determinism_max_vote([gb[p]["answers_by_run"][i]
                                   for i in shared])
        aa = mean(ga[p]["correct_by_run"][i] for i in shared)
        ab = mean(gb[p]["correct_by_run"][i] for i in shared)
        return da - db, aa - ab
    _pc_pairs = [_pc_pair(p) for p in pc]
    gaps_pc = [x[0] for x in _pc_pairs]
    gaps_pc_acc = [x[1] for x in _pc_pairs]
    pa, pb = cell_a.get("per_problem_all", {}), cell_b.get(
        "per_problem_all", {})
    common_all = sorted(set(pa) & set(pb))
    gaps_ef = [pa[p]["D_errfail"] - pb[p]["D_errfail"] for p in common_all]
    gaps_ef_acc = [pa[p]["acc_errfail"] - pb[p]["acc_errfail"]
                   for p in common_all]
    out = {
        "n": len(common),
        "D_gap_mean": round(mean(gaps_d), 4),
        "D_gap_ci": [round(x, 4) for x in bca_ci(gaps_d, BOOT_B, seed=seed)],
        "D_gap_perm_p": paired_permutation_p(gaps_d),
        "D_gap_strict_mean": round(mean(gaps_ds), 4),
        "D_gap_strict_ci": [round(x, 4) for x in
                            bca_ci(gaps_ds, BOOT_B, seed=seed + 2)],
        "D_gap_strict_perm_p": paired_permutation_p(gaps_ds),
        "acc_gap_mean": round(mean(gaps_acc), 4),
        "acc_gap_ci": [round(x, 4) for x in bca_ci(gaps_acc, BOOT_B,
                                                   seed=seed + 1)],
        "acc_gap_perm_p": paired_permutation_p(gaps_acc),
        "D_gap_ci_degenerate": len(set(gaps_d)) == 1,
        "acc_gap_ci_degenerate": len(set(gaps_acc)) == 1,
        "acc_strict_gap_mean": round(mean(gaps_accs), 4),
        "acc_strict_gap_ci": [round(x, 4) for x in
                              bca_ci(gaps_accs, BOOT_B, seed=seed + 4)],
        "acc_strict_gap_perm_p": paired_permutation_p(gaps_accs),
        "n_complete_only": len(complete),
        "D_gap_mean_complete_only": (round(mean(
            [ga[p]["D"] - gb[p]["D"] for p in complete]), 4)
            if complete else None),
        "D_gap_complete_only_ci": ([round(x, 4) for x in bca_ci(
            [ga[p]["D"] - gb[p]["D"] for p in complete], BOOT_B,
            seed=seed + 7)] if complete else None),
        "D_gap_complete_only_perm_p": (paired_permutation_p(
            [ga[p]["D"] - gb[p]["D"] for p in complete])
            if complete else None),
        "acc_gap_mean_complete_only": (round(mean(
            [ga[p]["acc"] - gb[p]["acc"] for p in complete]), 4)
            if complete else None),
        "acc_gap_complete_only_perm_p": (paired_permutation_p(
            [ga[p]["acc"] - gb[p]["acc"] for p in complete])
            if complete else None),
        "n_pairwise_complete": len(pc),
        "D_gap_pc_mean": round(mean(gaps_pc), 4) if gaps_pc else None,
        "D_gap_pc_ci": ([round(x, 4) for x in
                         bca_ci(gaps_pc, BOOT_B, seed=seed + 5)]
                        if gaps_pc else None),
        "D_gap_pc_perm_p": (paired_permutation_p(gaps_pc)
                            if gaps_pc else None),
        "acc_gap_pc_mean": round(mean(gaps_pc_acc), 4) if gaps_pc_acc
        else None,
        "acc_gap_pc_ci": ([round(x, 4) for x in
                           bca_ci(gaps_pc_acc, BOOT_B, seed=seed + 8)]
                          if gaps_pc_acc else None),
        "acc_gap_pc_perm_p": (paired_permutation_p(gaps_pc_acc)
                              if gaps_pc_acc else None),
        "n_errfail": len(common_all),
        "D_gap_errfail_mean": round(mean(gaps_ef), 4) if gaps_ef else None,
        "D_gap_errfail_ci": ([round(x, 4) for x in
                              bca_ci(gaps_ef, BOOT_B, seed=seed + 6)]
                             if gaps_ef else None),
        "D_gap_errfail_perm_p": (paired_permutation_p(gaps_ef)
                                 if gaps_ef else None),
        "acc_gap_errfail_mean": round(mean(gaps_ef_acc), 4) if gaps_ef_acc
        else None,
        "acc_gap_errfail_ci": ([round(x, 4) for x in
                                bca_ci(gaps_ef_acc, BOOT_B, seed=seed + 9)]
                               if gaps_ef_acc else None),
        "acc_gap_errfail_perm_p": (paired_permutation_p(gaps_ef_acc)
                                   if gaps_ef_acc else None),
    }
    if two_stage:
        # Paired two-stage bootstrap: resample problems, then resample RUN
        # INDICES once per drawn problem and apply the same indices to both
        # cells (runs are paired within a unit by construction).
        rng = random.Random(seed + 3)
        boots = []
        for _ in range(BOOT_B_2STAGE):
            drawn = rng.choices(common, k=len(common))
            gsum = 0.0
            for p in drawn:
                shared = sorted(set(ga[p]["answers_by_run"])
                                & set(gb[p]["answers_by_run"]))
                idx = [rng.choice(shared) for _ in shared]
                aa = [ga[p]["answers_by_run"][i] for i in idx]
                bb = [gb[p]["answers_by_run"][i] for i in idx]
                gsum += (determinism_max_vote(aa) - determinism_max_vote(bb))
            boots.append(gsum / len(drawn))
        boots.sort()
        lo = boots[max(0, int(0.025 * len(boots)))]
        hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
        out["D_gap_ci_2stage"] = [round(lo, 4), round(hi, 4)]
    return out


def paired_diff_within(cell: dict, other: dict, seed: int) -> dict:
    """Alias with clearer name for within-construction depth differences."""
    return paired_gap(cell, other, seed)


def analyze(loaded: dict) -> dict:
    chains = loaded["latest"]
    loaded_all = loaded.get("all", [])
    groups = sorted({c["_group"] for c in chains})
    out: dict = {"arms": {}, "primary": {}, "model_id_check": {},
                 "accounting": {}}
    _cells_cache: dict = {}
    primary_pvals: dict[str, float] = {}
    for group in groups:
        gchains = [c for c in chains if c["_group"] == group]
        model_ids = sorted({m for c in gchains
                            for m in c.get("model_ids_observed", []) if m})
        n_null_id_chains = sum(1 for c in gchains
                               if not c.get("model_ids_observed"))
        dev_chains = [c for c in gchains if c.get("deviations", 0) > 0]
        out["model_id_check"][group] = {
            "model_ids_observed": model_ids,
            "uniform_among_observed": len(model_ids) == 1,
            "n_chains_without_observed_id": n_null_id_chains,
            "deviation_chains": len(dev_chains),
            "deviation_steps": sum(c.get("deviations", 0) for c in gchains),
            "deviation_max_turns": max(
                (s.get("num_turns") or 0 for c in dev_chains
                 for s in c.get("steps", [])), default=0),
        }
        all_recs = [c for c in loaded["all"] if c["_group"] == group]
        accepted = [c for c in gchains if c["error"] is None]
        out["accounting"][group] = {
            "accepted_chains": len(accepted),
            "accepted_step_calls": sum(c["k"] for c in accepted),
            "all_chain_records": len(all_recs),
            "all_step_records": sum(len(c.get("steps", [])) for c in all_recs),
            "all_attempts": sum(
                (s.get("attempts") or 1)
                for c in all_recs for s in c.get("steps", [])),
            "errors_in_latest": sum(1 for c in gchains
                                    if c["error"] is not None),
        }
        out["arms"][group] = {}
        for cls in ("main", "ctl", "conf", "letters", "clean", "r5",
                    "r5letters"):
            cls_chains = [c for c in gchains if c["_class"] == cls]
            if not cls_chains:
                continue
            cells: dict = {}
            depths = sorted({c["k"] for c in cls_chains})
            # cache full cells (with per_problem) for cross-class estimators
            for cons in sorted({c["construction"] for c in cls_chains}):
                for k in depths:
                    sub = [c for c in cls_chains
                           if c["construction"] == cons and c["k"] == k]
                    if sub:
                        cells[f"{cons}_k{k}"] = cell_stats(sub)
                        _cells_cache[(group, cls,
                                      f"{cons}_k{k}")] = cells[f"{cons}_k{k}"]
            cls_out: dict = {
                "cells": {name: {k2: v2 for k2, v2 in cell.items()
                                 if k2 != "per_problem"}
                          for name, cell in cells.items()},
                "gaps": {}, "flatness": {}, "failure_mode": {},
            }
            max_k = max(depths)
            for k in depths:
                fk, dk = cells.get(f"fixed_k{k}"), cells.get(
                    f"degenerate_k{k}")
                if fk and dk:
                    cls_out["gaps"][f"fixed_minus_degenerate_k{k}"] = \
                        paired_gap(fk, dk, _seed_for("gap", group, cls, k),
                                   two_stage=(k == max_k))
            for pair in (("fixed", "finalonly"), ("finalonly", "degenerate"),
                         ("padded2", "degenerate"), ("fixed", "padded2"),
                         ("padded", "degenerate"), ("fixed", "padded")):
                a, b = pair
                ca, cb = cells.get(f"{a}_k{max_k}"), cells.get(f"{b}_k{max_k}")
                if ca and cb:
                    cls_out["gaps"][f"{a}_minus_{b}_k{max_k}"] = paired_gap(
                        ca, cb, _seed_for(a, b, group, cls, max_k),
                        two_stage=True)
            g1 = cls_out["gaps"].get("fixed_minus_degenerate_k1")
            if g1 and g1.get("n"):
                ci = g1.get("D_gap_ci") or [0, 0]
                aci = g1.get("acc_gap_ci") or [0, 0]
                g1["noise_rule_pass"] = bool(
                    abs(g1["D_gap_mean"]) <= 0.05
                    and ci[0] >= -0.10 and ci[1] <= 0.10
                    and abs(g1.get("acc_gap_mean") or 0) <= 0.05
                    and aci[0] >= -0.10 and aci[1] <= 0.10)
            if "fixed_k1" in cells:
                for k in depths:
                    if k != 1 and f"fixed_k{k}" in cells:
                        cls_out["flatness"][f"k1_minus_k{k}"] = paired_gap(
                            cells["fixed_k1"], cells[f"fixed_k{k}"],
                            _seed_for("flat", group, cls, k))
            # Pre-registered rebound estimand (R2 endpoint 4).
            if ("degenerate_k7" in cells and "degenerate_k5" in cells):
                cls_out["rebound_deg_k7_minus_k5"] = paired_gap(
                    cells["degenerate_k7"], cells["degenerate_k5"],
                    _seed_for("rebound", group, cls))
            for cons in ("degenerate", "fixed", "finalonly", "padded2",
                         "padded"):
                cell = cells.get(f"{cons}_k{max_k}")
                if not cell:
                    continue
                rows = [{"problem_id": p, "D": round(v["D"], 3),
                         "modal_answer": v["modal_answer"],
                         "modal_correct": v["modal_correct"],
                         "acc": round(v["acc"], 3)}
                        for p, v in sorted(cell["per_problem"].items())]
                high = [r for r in rows if r["D"] >= CONCENTRATION_D]
                cls_out["failure_mode"][cons] = {
                    "k": max_k, "rows": rows, "n_highD": len(high),
                    "n_highD_modal_wrong": sum(
                        1 for r in high if not r["modal_correct"]),
                }
            # TOST equivalence for finalonly vs fixed (frozen R3 endpoint 3):
            # pass iff the 90% BCa CI of the paired gap lies within ±0.10.
            fo = cls_out["gaps"].get(f"fixed_minus_finalonly_k{max_k}")
            if fo and fo.get("n"):
                fx = cells.get(f"fixed_k{max_k}")
                fl = cells.get(f"finalonly_k{max_k}")
                common = sorted(set(fx["per_problem"]) & set(fl["per_problem"]))
                gaps90 = [fx["per_problem"][p]["D"] - fl["per_problem"][p]["D"]
                          for p in common]
                lo90, hi90 = bca_ci(gaps90, BOOT_B, alpha=0.10,
                                    seed=_seed_for("tost", group, cls))
                fo["tost_margin"] = 0.10
                fo["tost_ci90"] = [round(lo90, 4), round(hi90, 4)]
                if len(set(gaps90)) == 1:
                    fo["tost_pass"] = None
                    fo["tost_note"] = ("degenerate gap distribution; CI "
                                       "unavailable; descriptive only "
                                       "(Amendment A6)")
                else:
                    fo["tost_pass"] = equivalence_decision(lo90, hi90, 0.10)
                    if fo["tost_pass"] is None:
                        fo["tost_note"] = "equivalence-margin boundary contact; inconclusive"
            out["arms"][group][cls] = cls_out
            prim = cls_out["gaps"].get(f"fixed_minus_degenerate_k{max_k}")
            if prim and prim.get("n"):
                if cls == "conf":       # confirmatory wave = paper primary
                    primary_pvals[group] = prim["D_gap_perm_p"]
                    out["primary"][group] = {"k": max_k, **prim}
                elif cls == "main":     # exploratory (R1+R2) estimates
                    out.setdefault("primary_exploratory", {})[group] = \
                        {"k": max_k, **prim}
    # Fall back to exploratory as primary only for groups with no conf data.
    for group, prim in out.get("primary_exploratory", {}).items():
        if group not in out["primary"]:
            primary_pvals[group] = prim["D_gap_perm_p"]
            out["primary"][group] = dict(prim)
    if primary_pvals:
        adjusted = holm_correct(primary_pvals)
        for group, p_adj in adjusted.items():
            out["primary"][group]["D_gap_perm_p_holm"] = p_adj
    # Per-class, per-metric Holm over the PRE-REGISTERED family (Amendment
    # A6): the family is the class's named arms; missing arms mark the
    # family incomplete — the family is never silently shrunk.
    metric_keys = {"D_gap_perm_p": "D_gap_perm_p_holm_cls",
                   "acc_gap_perm_p": "acc_gap_perm_p_holm_cls",
                   "D_gap_strict_perm_p": "D_gap_strict_perm_p_holm_cls",
                   "acc_strict_gap_perm_p": "acc_strict_gap_perm_p_holm_cls"}
    PANEL_N = {"main": 24, "ctl": 12, "conf": 12, "letters": 12,
               "clean": 12, "r5": 12, "r5letters": 12}
    out["family_status"] = {}
    for cls in ("main", "ctl", "conf", "letters", "clean", "r5",
                "r5letters"):
        need_n = PANEL_N.get(cls, 12)
        regime = REGIME.get(cls, "a6")
        n_key = ("n_pairwise_complete" if regime == "a8" else "n")

        def _grid_ok(g):
            cells_g = (out["arms"].get(g, {}).get(cls, {})
                       .get("cells", {}))
            for cname in GRIDS.get(cls, []):
                cell = cells_g.get(cname)
                if not cell or cell.get("n_problems_used") != need_n:
                    return False
            if regime == "a8":
                gaps_g = (out["arms"].get(g, {}).get(cls, {})
                          .get("gaps", {}))
                for gap_name in ("fixed_minus_degenerate_k7",
                                 "fixed_minus_degenerate_k1",
                                 "fixed_minus_finalonly_k7"):
                    if gap_name == "fixed_minus_finalonly_k7" and \
                            "finalonly_k7" not in GRIDS.get(cls, []):
                        continue
                    gg = gaps_g.get(gap_name)
                    if not gg or gg.get("n_pairwise_complete") != need_n:
                        return False
            return True

        present = [g for g in groups
                   if (out["arms"].get(g, {}).get(cls, {})
                       .get("gaps", {}).get("fixed_minus_degenerate_k7",
                                            {}).get(n_key) == need_n)
                   and _grid_ok(g)]
        # Over-full family (A11/A12): data from a non-registered arm in a
        # registered class voids the family.
        rogue = [g for g in groups
                 if g not in (FAMILIES.get(cls) or [g])
                 and out["arms"].get(g, {}).get(cls)]
        partial = [g for g in groups if g not in present
                   and (out["arms"].get(g, {}).get(cls, {})
                        .get("gaps", {}).get("fixed_minus_degenerate_k7",
                                             {}).get("n"))]
        expected = FAMILIES.get(cls)
        m_total = len(expected) if expected else None
        meta_only = any(
            m.get("record_type") == "meta"
            and (("R4CLEAN" in str(m.get("campaign")) and cls == "clean")
                 or (str(m.get("campaign")).endswith("-R5")
                     and cls == "r5")
                 or (str(m.get("campaign")).endswith("-R5LET")
                     and cls == "r5letters"))
            for m in loaded_all)
        if expected is not None and not present and not partial and meta_only:
            out["family_status"][cls] = {
                "expected_arms": expected, "present_arms": [],
                "partial_arms": [], "complete": False, "success": False,
                "missing_arms": expected,
                "note": "wave attempted (meta records) but no usable data"}
        if expected is not None and (present or partial):
            missing = sorted(set(expected) - set(present))
            out["family_status"][cls] = {
                "expected_arms": expected, "present_arms": present,
                "partial_arms": partial,
                "complete": not missing and not rogue,
                "missing_arms": missing}
            if rogue:
                out["family_status"][cls]["rogue_arms"] = rogue
        mk = (dict(metric_keys) if regime == "a6" else
              {"D_gap_pc_perm_p": "D_gap_pc_perm_p_holm_cls",
               "acc_gap_pc_perm_p": "acc_gap_pc_perm_p_holm_cls",
               "D_gap_strict_perm_p": "D_gap_strict_perm_p_holm_cls",
               "acc_strict_gap_perm_p": "acc_strict_gap_perm_p_holm_cls"})
        for pkey, hkey in mk.items():
            pv = {}
            fam = (expected if expected is not None else present + partial)
            for group in [g for g in present + partial if g in fam]:
                g = out["arms"][group][cls]["gaps"][
                    "fixed_minus_degenerate_k7"]
                if g.get(pkey) is not None:
                    pv[group] = g[pkey]
            if pv:
                adj = holm_correct(pv, m_total=m_total)
                for group, p_adj in adj.items():
                    out["arms"][group][cls]["gaps"][
                        "fixed_minus_degenerate_k7"][hkey] = p_adj
        # Registered success rule (Amendment A7): family complete AND every
        # expected arm has D and acc Holm-adj p<0.01 with CIs excluding 0.
        fs = out["family_status"].get(cls)
        if fs is not None:
            ok = fs["complete"]
            for g in (expected or []):
                gg = (out["arms"].get(g, {}).get(cls, {})
                      .get("gaps", {}).get("fixed_minus_degenerate_k7", {}))
                # Directional (fixed-advantage) success: lower CI bound > 0.
                pairs = ((("D_gap_perm_p_holm_cls", "D_gap_ci"),
                          ("acc_gap_perm_p_holm_cls", "acc_gap_ci"))
                         if regime == "a6" else
                         (("D_gap_pc_perm_p_holm_cls", "D_gap_pc_ci"),
                          ("acc_gap_pc_perm_p_holm_cls", "acc_gap_pc_ci")))
                for hk, cik in pairs:
                    ci = gg.get(cik)
                    if not (gg.get(hk) is not None and gg[hk] < 0.01 and ci
                            and ci[0] > 0):
                        ok = False
                g1 = (out["arms"].get(g, {}).get(cls, {})
                      .get("gaps", {}).get("fixed_minus_degenerate_k1", {}))
                # Missing or reduced-panel k=1 fails the gate (A9).
                if g1.get("n") != need_n or not g1.get("noise_rule_pass"):
                    ok = False
                    fs.setdefault("k1_gate_failures", []).append(g)
                # Provenance-complete chains required for waves collected
                # under Amendment A6+ (R4CLEAN, R5, R5LET). R3-era waves
                # (conf, letters) are governed by their own registered
                # mechanism (frozen commit + fail-closed identity) and are
                # not retroactively re-judged.
                if cls in ("conf", "letters"):
                    fs.setdefault("provenance_model",
                                  "R3-era: frozen commit + fail-closed "
                                  "identity; per-chain invocation ids "
                                  "introduced by A6 postdate this wave")
                if cls in ("clean", "r5", "r5letters"):
                    gch = [c for c in chains if c["_group"] == g
                           and c["_class"] == cls]
                    # Identity validation (A12): every chain's observed
                    # model must equal the registered pin; PENDING never
                    # validates.
                    exp_model, _exp_eff = EXPECTED_MODELS.get(g, (None, None))
                    ids_all = {m for c in gch
                               for m in (c.get("model_ids_observed") or
                                         [None])}
                    if (exp_model is None or "PENDING" in str(exp_model)
                            or ids_all != {exp_model}):
                        ok = False
                        fs.setdefault("identity_failures", []).append(
                            {"arm": g, "expected": exp_model,
                             "observed": sorted(str(x) for x in ids_all)})
                    # Randomization audit (A12): unit_order must be present
                    # and the recorded order_position must be consistent.
                    if any(not c.get("unit_order")
                           or c.get("order_position") is None
                           or (c.get("construction") not in
                               (c.get("unit_order") or []))
                           for c in gch):
                        ok = False
                        fs.setdefault("order_audit_failures", []).append(g)
                    metas = [m for m in loaded_all
                             if m.get("record_type") == "meta"
                             and m.get("_group") == g
                             and m.get("invocation_id")]
                    meta_by_inv = {m["invocation_id"]: m for m in metas}
                    bad = False
                    fps = set()
                    for c in gch:
                        inv, fp = c.get("invocation_id"), c.get("config_fp")
                        m = meta_by_inv.get(inv)
                        fps.add(fp)
                        if (inv is None or fp is None or m is None
                                or m.get("config_fp") != fp
                                or not m.get("git_tree_clean")):
                            bad = True
                            break
                    # One experimental configuration per arm-class (A11):
                    # resumed invocations are fine, config changes are not.
                    if len(fps) > 1:
                        bad = True
                        fs.setdefault("mixed_config_arms", []).append(g)
                    # Registered panel + run-set validation (A11)
                    pids = {c["problem_id"] for c in gch}
                    runs_seen = {c.get("run_idx") for c in gch}
                    reg = REGISTERED_PANELS.get(
                        "letters" if cls in ("r5letters", "letters")
                        else "gsm8k_base")
                    if reg is not None and (pids != reg
                                            or runs_seen != {0, 1, 2, 3, 4}):
                        bad = True
                        fs.setdefault("panel_mismatch_arms", []).append(g)
                    if bad:
                        ok = False
                        fs.setdefault("provenance_gaps", []).append(g)
                    aborts = [m for m in loaded_all
                              if m.get("record_type") == "meta_abort"
                              and m.get("arm", "").startswith(g)]
                    if aborts:
                        ok = False
                        fs["complete"] = False
                        fs.setdefault("aborted_arms", []).append(g)
                if cls == "clean":
                    gates = [(m.get("scaffold_status")
                              or (m.get("scaffold_probe") or {}).get("gate"))
                             for m in loaded_all
                             if m.get("record_type") == "meta"
                             and m.get("campaign") in CLEAN_CAMPAIGNS
                             and m.get("_group") == g]
                    if "REDUCED_SCAFFOLD" in gates:
                        ok = False
                        fs["relabel"] = "reduced-scaffold"
                    elif "PASS" not in gates:
                        ok = False
                        fs.setdefault("scaffold_gate_missing", []).append(g)
            fs["success"] = ok
    # Registered clean-vs-conf descriptive estimator (Amendment A6/A8):
    # per-problem paired diff clean − conf per (construction, k) cell.
    for group in groups:
        gcls = out["arms"].get(group, {})
        if "clean" in gcls and "conf" in gcls:
            comp = {}
            for cname in gcls["clean"]["cells"]:
                a = _cells_cache.get((group, "clean", cname))
                b = _cells_cache.get((group, "conf", cname))
                if a and b:
                    comp[cname] = paired_gap(
                        a, b, _seed_for("cleanconf", group, cname))
            if comp:
                gcls["clean_vs_conf_descriptive"] = {
                    k2: {kk: vv for kk, vv in v.items()}
                    for k2, v in comp.items()}
    # Unit-integrity: within frozen classes, all constructions of a unit
    # must share one invocation (legacy no-id records exempt).
    integ: dict = {}
    units: dict = {}
    for c in chains:
        if c.get("invocation_id") is None:
            continue
        units.setdefault((c["_group"], c["_class"], c["k"],
                          c["problem_id"], c["run_idx"]), set()).add(
            c["invocation_id"])
    for (g, cls, *_), invs in units.items():
        if len(invs) > 1:
            integ[cls] = integ.get(cls, 0) + 1
    out["unit_integrity_violations"] = integ
    for cls, cnt in integ.items():
        if cls in out.get("family_status", {}):
            out["family_status"][cls]["complete"] = False
            out["family_status"][cls]["success"] = False
            out["family_status"][cls]["mixed_invocation_units"] = cnt
    return out


def write_refusal_examples(chains: list[dict], results_dir: Path,
                           limit: int = 12) -> None:
    examples = []
    for c in chains:
        if c.get("any_step_refusal") and c["construction"] == "degenerate":
            step = next((s for s in c["steps"] if s.get("refusal")), None)
            if step:
                examples.append({
                    "arm": c["_group"], "k": c["k"],
                    "problem_id": c["problem_id"], "run_idx": c["run_idx"],
                    "excerpt": step["output"][:280],
                })
        if len(examples) >= limit:
            break
    (results_dir / "refusal_examples.json").write_text(
        json.dumps(examples, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--include-smoke", action="store_true")
    args = ap.parse_args()
    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_chains(Path(args.runs_dir),
                         include_smoke=args.include_smoke)
    if not loaded["latest"]:
        print("No chain records found.")
        return 1
    summary = analyze(loaded)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_refusal_examples(loaded["latest"], results_dir)
    print(f"Analyzed {len(loaded['latest'])} latest chains "
          f"({len(loaded['all'])} records) -> {results_dir}/summary.json")
    for group, prim in summary.get("primary", {}).items():
        print(f"  PRIMARY {group}: D gap@k{prim['k']} = "
              f"{prim['D_gap_mean']:+.3f} CI {prim['D_gap_ci']} "
              f"2stage {prim.get('D_gap_ci_2stage')} "
              f"strict {prim['D_gap_strict_mean']:+.3f} "
              f"holm={prim.get('D_gap_perm_p_holm'):.4f} n={prim['n']}")
        ctl = summary["arms"][group].get("ctl", {}).get("gaps", {})
        fo = ctl.get("fixed_minus_finalonly_k7")
        if fo and fo.get("n"):
            print(f"    ctl fixed-finalonly: {fo['D_gap_mean']:+.3f} "
                  f"CI {fo['D_gap_ci']}  finalonly-degen: "
                  f"{ctl['finalonly_minus_degenerate_k7']['D_gap_mean']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
