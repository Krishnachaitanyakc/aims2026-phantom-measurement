"""Offline audit for the narrow AIMS revision; never modifies source evidence."""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifact-root", type=Path, default=PACKAGE / "artifact")
parser.add_argument("--summary", type=Path)
parser.add_argument(
    "--out", type=Path, default=PACKAGE / "evidence/camera_ready_evidence.json"
)
args = parser.parse_args()
ROOT = args.artifact_root.resolve()
summary_path = args.summary or ROOT / "results/summary.json"
summary = json.loads(summary_path.read_text())
args.out.parent.mkdir(parents=True, exist_ok=True)
panel = json.loads((ROOT / "data/subset_seed20260728.json").read_text())
problems = {p["id"]: p for p in panel["problems"]}
parsing_tree = ast.parse((ROOT / "harness/parsing.py").read_text())
refusal_patterns = next(
    ast.literal_eval(node.value)
    for node in parsing_tree.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "REFUSAL_PATTERNS"
        for target in node.targets
    )
)


def same(a, b):
    if not a or not b:
        return False
    try:
        fa = float(a.replace("$", "").replace(",", "").rstrip("%"))
        fb = float(b.replace("$", "").replace(",", "").rstrip("%"))
        return abs(fa - fb) <= 1e-6 * max(1, abs(fa), abs(fb))
    except ValueError:
        return a.strip() == b.strip()


def modal_count(answers):
    groups = []
    for a in sorted(answers):
        group = next((g for g in groups if same(g[0], a)), None)
        if group is None:
            groups.append([a])
        else:
            group.append(a)
    return max(map(len, groups))


def sign_p(gaps):
    """Literal enumeration of every sign assignment; exact Fraction arithmetic."""
    observed = abs(sum(gaps))
    exceed = sum(
        abs(sum(s * g for s, g in zip(signs, gaps))) >= observed
        for signs in product([-1, 1], repeat=len(gaps))
    )
    return Fraction(exceed, 2 ** len(gaps))


def holm(ps):
    answer = {}
    bound = Fraction(0)
    for i, (key, value) in enumerate(sorted(ps.items(), key=lambda x: x[1])):
        bound = min(Fraction(1), max(bound, (len(ps) - i) * value))
        answer[key] = bound
    return answer


def clean(tok):
    return tok.replace("$", "").replace(",", "").rstrip("%").strip()


def parse(text):
    markers = re.findall(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text)
    tokens = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?%?", text)
    return (
        (clean(markers[-1]), "hash")
        if markers
        else ((clean(tokens[-1]), "fallback") if tokens else ("", "none"))
    )


def rounded(frac):
    return round(float(frac), 4)


evidence = {
    "scope": "SW-JUL26-R3 arithmetic only; three models, 12 problems, five chains per cell",
    "audit_status": "Every included cell and k7 contrast independently verified against raw logs",
    "source_dataset_sha256": hashlib.sha256(
        (ROOT / "data/gsm8k_test.jsonl").read_bytes()
    ).hexdigest(),
    "source_dataset_pin_matches": hashlib.sha256(
        (ROOT / "data/gsm8k_test.jsonl").read_bytes()
    ).hexdigest()
    == panel["source_sha256"],
    "models": {},
    "method": {
        "strict_agreement": "Each non-hash or refusal final output becomes a distinct no-answer, still in denominator; N=5 floor is 0.2.",
        "inference": "Exact two-sided sign-flip enumeration of 4096 assignments; inferential validity requires a sign-symmetry/exchangeability null.",
        "intervals": "Retained from the supplied regenerated analysis after independently verifying its raw input cells and contrast estimates; problem-cluster BCa 95% with 10,000 draws; two-stage intervals are sensitivity analyses.",
        "cli_versions": "Protocol documents Claude Code 2.1.220 and Codex 0.145.0; versions not logged in per-call R3 records.",
        "decoding": "CLI defaults; temperature and sampling seed unexposed, GPT medium reasoning effort.",
    },
    "cautions": [
        "R3 plan timing is not independently public-verifiable; call it a frozen-plan replication.",
        "Do not claim general model-family ranks, chain capability, deterministic-model behavior, or defect prevalence.",
        "Do not state TOST equivalence; Haiku/Sonnet have degenerate bootstrap gap distributions and GPT touches the margin.",
        "Rebound is a descriptive/parser-sensitivity observation, not multiplicity-adjusted confirmatory significance.",
        "No reviewer-verifiable historical 14–42x causal comparison survives.",
    ],
}
all_counts = Counter()
pvals = defaultdict(dict)
for model, filename in [
    ("haiku", "haiku-r3.jsonl"),
    ("codex-gpt56", "codex-r3.jsonl"),
    ("sonnet", "sonnet-r3.jsonl"),
]:
    path = ROOT / "runs" / filename
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    raw = [r for r in records if r.get("record_type") == "chain"]
    latest = {}
    for r in raw:
        assert r["campaign"] == "SW-JUL26-R3"
        key = tuple(
            r[k]
            for k in (
                "arm_group",
                "campaign",
                "construction",
                "k",
                "problem_id",
                "run_idx",
            )
        )
        latest[key] = r
    chains = list(latest.values())
    counts = {
        "selected_chains": len(chains),
        "selected_step_calls": sum(len(r["steps"]) for r in chains),
        "stored_chain_records": len(raw),
        "stored_step_objects": sum(len(r["steps"]) for r in raw),
        "inference_attempts": sum(
            s.get("attempts") or 1 for r in raw for s in r["steps"]
        ),
        "error_chains": sum(r["error"] is not None for r in chains),
        "deviation_chains": sum(bool(r["deviations"]) for r in chains),
    }
    all_counts.update(counts)
    checks = Counter()
    cells = defaultdict(list)
    for r in chains:
        cells[f"{r['construction']}_k{r['k']}"].append(r)
        for i, s in enumerate(r["steps"]):
            assert (
                hashlib.sha256(s["prompt"].encode()).hexdigest() == s["prompt_sha256"]
            )
            assert parse(s["output"]) == (s["parsed_answer"], s["parse_path"])
            assert s["refusal"] == any(
                p in s["output"].lower() for p in refusal_patterns
            )
            checks["verified_step_hashes_parses_and_refusal_flags"] += 1
            present = problems[r["problem_id"]]["question"] in s["prompt"]
            expected = (
                r["construction"] == "fixed"
                or i == 0
                or (r["construction"] == "finalonly" and i == r["k"] - 1)
            )
            assert present == expected, (model, r["construction"], i)
            checks["verified_task_presence_steps"] += 1
        assert parse(r["steps"][-1]["output"]) == (r["answer"], r["parse_path"])
        assert r["final_refusal"] == r["steps"][-1]["refusal"]
        assert r["any_step_refusal"] == any(s["refusal"] for s in r["steps"])
        assert same(r["answer"], r["gold"]) == r["correct"]
        checks["verified_final_parses_and_correctness"] += 1
    model_out = {
        "source_log": path.relative_to(ROOT).as_posix(),
        "source_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "counts": counts,
        "checks": dict(checks),
        "observed_model_ids": sorted(
            {x for r in chains for x in r["model_ids_observed"]}
        ),
        "selected_first_call_utc": min(s["ts"] for r in chains for s in r["steps"]),
        "selected_last_call_utc": max(s["ts"] for r in chains for s in r["steps"]),
        "cells": {},
        "contrasts": {},
    }
    per_cell = {}
    for name, rs in sorted(cells.items()):
        by_problem = defaultdict(list)
        for r in rs:
            by_problem[r["problem_id"]].append(r)
        assert set(by_problem) == set(problems)
        per = {}
        for pid, prs in sorted(by_problem.items()):
            assert len(prs) == 5 and {r["run_idx"] for r in prs} == set(range(5))
            per[pid] = {
                "D": Fraction(modal_count([r["answer"] for r in prs]), 5),
                "D_marker_only": Fraction(
                    modal_count(
                        [r["answer"] if r["parse_path"] == "hash" else "" for r in prs]
                    ),
                    5,
                ),
                "D_refusal_only": Fraction(
                    modal_count(
                        [r["answer"] if not r["final_refusal"] else "" for r in prs]
                    ),
                    5,
                ),
                "D_strict": Fraction(
                    modal_count(
                        [
                            r["answer"]
                            if r["parse_path"] == "hash" and not r["final_refusal"]
                            else ""
                            for r in prs
                        ]
                    ),
                    5,
                ),
                "acc": Fraction(sum(r["correct"] for r in prs), 5),
                "acc_strict": Fraction(
                    sum(
                        r["correct"]
                        and r["parse_path"] == "hash"
                        and not r["final_refusal"]
                        for r in prs
                    ),
                    5,
                ),
            }
        per_cell[name] = per
        means = {
            k: rounded(sum(p[k] for p in per.values()) / 12)
            for k in ("D", "D_strict", "acc", "acc_strict")
        }
        original = summary["arms"][model]["conf"]["cells"][name]
        assert all(v == original[k + "_mean"] for k, v in means.items())
        means.update(
            {
                k: rounded(sum(p[k] for p in per.values()) / 12)
                for k in ("D_marker_only", "D_refusal_only")
            }
        )
        fb = [r for r in rs if r["parse_path"] == "fallback"]
        model_out["cells"][name] = {
            "n_problems": 12,
            "n_runs_per_problem": 5,
            "n_chains": len(rs),
            **means,
            "correct_count": sum(r["correct"] for r in rs),
            "strict_correct_count": sum(
                r["correct"] and r["parse_path"] == "hash" and not r["final_refusal"]
                for r in rs
            ),
            "parse_counts": dict(Counter(r["parse_path"] for r in rs)),
            "final_refusal_count": sum(r["final_refusal"] for r in rs),
            "fallback_values": dict(Counter(r["answer"] for r in fb)),
            "final_or_previous_step_index_fallback_count": sum(
                r["answer"] in (str(r["k"]), str(r["k"] - 1)) for r in fb
            ),
            "modal_votes_total": int(sum(p["D"] * 5 for p in per.values())),
            "strict_modal_votes_total": int(
                sum(p["D_strict"] * 5 for p in per.values())
            ),
        }
    for name, ca, cb in [
        ("fixed_minus_degenerate_k7", "fixed_k7", "degenerate_k7"),
        ("fixed_minus_finalonly_k7", "fixed_k7", "finalonly_k7"),
        ("rebound_deg_k7_minus_k5", "degenerate_k7", "degenerate_k5"),
    ]:
        est = {}
        for metric in ("D", "D_strict", "acc", "acc_strict"):
            gaps = [
                per_cell[ca][p][metric] - per_cell[cb][p][metric]
                for p in sorted(problems)
            ]
            est[metric] = {
                "mean_gap": rounded(sum(gaps) / 12),
                "exact_gap": str(sum(gaps) / 12),
                "per_problem_gaps": [str(g) for g in gaps],
                "sign_flip_p": float(sign_p(gaps)),
                "sign_flip_p_exact": str(sign_p(gaps)),
            }
            if name == "fixed_minus_degenerate_k7":
                pvals[metric][model] = sign_p(gaps)
        orig = (
            summary["arms"][model]["conf"][name]
            if name.startswith("rebound")
            else summary["arms"][model]["conf"]["gaps"][name]
        )
        for metric, prefix in [
            ("D", "D_gap"),
            ("D_strict", "D_gap_strict"),
            ("acc", "acc_gap"),
            ("acc_strict", "acc_strict_gap"),
        ]:
            assert est[metric]["mean_gap"] == orig[prefix + "_mean"]
            assert est[metric]["sign_flip_p"] == orig[prefix + "_perm_p"]
        est["reproduced_intervals"] = {
            k: v
            for k, v in orig.items()
            if k
            in (
                "D_gap_ci",
                "D_gap_ci_2stage",
                "D_gap_strict_ci",
                "acc_gap_ci",
                "acc_strict_gap_ci",
                "tost_ci90",
                "tost_pass",
            )
        }
        model_out["contrasts"][name] = est
    evidence["models"][model] = model_out
for metric, ps in pvals.items():
    for model, p in holm(ps).items():
        ev = evidence["models"][model]["contrasts"]["fixed_minus_degenerate_k7"][metric]
        ev["holm_p_across_three_models"] = float(p)
        key = {
            "D": "D_gap_perm_p_holm_cls",
            "D_strict": "D_gap_strict_perm_p_holm_cls",
            "acc": "acc_gap_perm_p_holm_cls",
            "acc_strict": "acc_strict_gap_perm_p_holm_cls",
        }[metric]
        assert (
            float(p)
            == summary["arms"][model]["conf"]["gaps"]["fixed_minus_degenerate_k7"][key]
        )
evidence["counts_total"] = dict(all_counts)
args.out.write_text(json.dumps(evidence, indent=2) + "\n")
print(
    json.dumps(
        {
            "audit_status": evidence["audit_status"],
            "counts_total": evidence["counts_total"],
            "models": {
                k: {
                    "counts": v["counts"],
                    "checks": v["checks"],
                    "dates": [
                        v["selected_first_call_utc"],
                        v["selected_last_call_utc"],
                    ],
                }
                for k, v in evidence["models"].items()
            },
        },
        indent=2,
    )
)
