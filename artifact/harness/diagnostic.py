"""Construction-consistency checks for the supplied logs.

Four checks, each exposed as a CLI subcommand:

1. `check-prompts`  — problem-persistence assertion: the canonicalized task
   statement must be a substring of every step's rendered prompt.
   Input: flat JSONL of {"chain_id", "step", "prompt", "problem"}, or
   native chain logs with --problems pointing to the supplied problem panel.
2. `gate-output`    — refusal gate: flag refusal-shaped model output BEFORE
   answer parsing so a numeric fallback never silently scores a refusal.
   Input: text on stdin or --file.
3. `audit-log`      — failure-mode-concentration audit: per (construction,
   depth) cell computes determinism D, accuracy, and refusal/fallback mix
   from a chain-record JSONL (this harness's format or any file providing
   construction/k/problem_id/answer/correct), and flags the red-flag
   signature "D rising while accuracy falls with depth".
4. `calibrate`      — dual-construction calibration: given chain-record
   JSONLs for a production construction and a context-preserving reference
   on the same problems, reports the paired D and accuracy gaps; a large
   gap flags construction sensitivity for inspection. (Data collection itself is
   an evaluation-harness concern; this check consumes the resulting logs.
   Cost formula for a minimal run: n_problems x N_runs x 2 constructions x
   (sum of chain depths) LM calls — e.g. 12 x 3 x 2 x (1+7) = 576.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

from .metrics import determinism_max_vote
from .parsing import is_refusal


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def problem_persists(prompt: str, problem: str) -> bool:
    return bool(canonicalize(problem)) and canonicalize(problem) in canonicalize(prompt)


def check_prompt_records(records: list[dict]) -> dict:
    failures = []
    for rec in records:
        if not problem_persists(rec["prompt"], rec["problem"]):
            failures.append({"chain_id": rec.get("chain_id"),
                             "step": rec.get("step")})
    return {"n_steps_checked": len(records),
            "n_failures": len(failures),
            "failures": failures[:50],
            "passed": bool(records) and not failures}


def load_prompt_records(path: Path, problems_path: Path | None = None,
                        construction: str | None = None) -> list[dict]:
    """Adapt retained native chains to the same flat prompt check.

    Problem text is supplied by the pinned panel, never inferred from the
    prompt being checked. Latest-record selection matches the log audit.
    """
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if not any("steps" in r for r in records):
        if construction is not None:
            raise ValueError("--construction requires native chain logs")
        return records
    if problems_path is None:
        raise ValueError("native chain logs require --problems")
    panel = json.loads(problems_path.read_text())
    problems = {r["id"]: r["question"] for r in panel["problems"]}
    chains = _load_chain_records(path)
    if construction is not None:
        chains = [r for r in chains if r["construction"] == construction]
    if not chains:
        raise ValueError("no chains selected")
    _single_namespace(chains)
    flat = []
    for chain in chains:
        problem_id = chain["problem_id"]
        if problem_id not in problems:
            raise ValueError(f"unknown problem in native log: {problem_id}")
        chain_id = ":".join(map(str, (chain.get("campaign", ""),
                            chain.get("arm_group", chain.get("arm", "")),
                            chain["construction"], chain["k"], problem_id,
                            chain["run_idx"])))
        for step_no, step in enumerate(chain.get("steps", []), 1):
            flat.append({"chain_id": chain_id, "step": step_no,
                         "prompt": step["prompt"],
                         "problem": problems[problem_id]})
    return flat


def refusal_gate(output: str) -> dict:
    flagged = is_refusal(output)
    return {"refusal": flagged, "safe_to_parse": not flagged}


def _load_chain_records(path: Path) -> list[dict]:
    """Native-log-safe loader: keeps only the LATEST record per logical key
    (superseded re-collections in this harness's logs are deduplicated the
    same way the analysis pipeline does)."""
    latest: dict[tuple, dict] = {}
    order = 0
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path.name}") from exc
        if r.get("record_type", "chain") == "chain" and "answer" in r:
            key = (r.get("campaign", ""),
                   r.get("arm_group", r.get("arm", "")),
                   r.get("construction", "default"), r.get("k", 0),
                   r.get("problem_id"), r.get("run_idx", order))
            latest[key] = r
            order += 1
    return list(latest.values())


def _single_namespace(records: list[dict]) -> tuple:
    namespaces = {(r.get("campaign", ""),
                   r.get("arm_group", r.get("arm", "")),
                   tuple(sorted(r.get("model_ids_observed", []))))
                  for r in records}
    if len(namespaces) != 1:
        raise ValueError("select one nonempty model/campaign namespace per input")
    return next(iter(namespaces))


def _cells(recs: list[dict]) -> dict:
    cells: dict = {}
    for r in recs:
        if r.get("error"):
            continue
        key = (r.get("construction", "default"), r.get("k", 0))
        cells.setdefault(key, {}).setdefault(
            r["problem_id"], []).append(r)
    out = {}
    for (cons, k), by_p in sorted(cells.items()):
        ds = [determinism_max_vote([x["answer"] for x in v])
              for v in by_p.values() if len(v) >= 2]
        accs = [mean(1.0 if x.get("correct") else 0.0 for x in v)
                for v in by_p.values()]
        flat = [x for v in by_p.values() for x in v]
        n = len(flat) or 1
        out[f"{cons}_k{k}"] = {
            "construction": cons, "k": k,
            "D": round(mean(ds), 4) if ds else None,
            "acc": round(mean(accs), 4) if accs else None,
            "n_problems": len(by_p),
            "refusal_rate": round(sum(
                1 for x in flat
                if x.get("any_step_refusal") or x.get("final_refusal")) / n, 4),
            "fallback_rate": round(sum(
                1 for x in flat if x.get("parse_path") == "fallback") / n, 4),
        }
    return out


def audit_log(path: Path) -> dict:
    records = _load_chain_records(path)
    _single_namespace(records)
    cells = _cells(records)
    warnings = []
    by_cons: dict = {}
    for cell in cells.values():
        by_cons.setdefault(cell["construction"], []).append(cell)
    for cons, arr in by_cons.items():
        arr.sort(key=lambda c: c["k"])
        for prev, cur in zip(arr, arr[1:]):
            if (prev["D"] is not None and cur["D"] is not None
                    and cur["D"] > prev["D"] + 1e-9
                    and cur["acc"] is not None and prev["acc"] is not None
                    and cur["acc"] < prev["acc"] - 1e-9):
                warnings.append(
                    f"FAILURE-MODE-CONCENTRATION signature in "
                    f"'{cons}': D rises {prev['D']:.2f}->{cur['D']:.2f} "
                    f"while accuracy falls {prev['acc']:.2f}->"
                    f"{cur['acc']:.2f} (k={prev['k']}->{cur['k']})")
    return {"cells": cells, "warnings": warnings,
            "passed": not warnings}


def calibrate(prod_path: Path, ref_path: Path,
              gap_threshold: float = 0.15,
              prod_construction: str | None = None,
              ref_construction: str | None = None) -> dict:
    """PAIRED dual-construction calibration: gaps are computed per common
    (problem, depth) over runs shared by both logs, then averaged; depths or
    problems present in only one log are reported, not silently dropped."""
    if not 0 <= gap_threshold <= 1:
        raise ValueError("gap_threshold must be in [0, 1]")

    def select(path, construction):
        recs = _load_chain_records(path)
        if construction is not None:
            recs = [r for r in recs if r.get("construction") == construction]
        namespace = _single_namespace(recs)
        if len({r.get("construction", "default") for r in recs}) != 1:
            raise ValueError("mixed constructions: select a construction explicitly")
        if any("run_idx" not in r for r in recs):
            raise ValueError("calibration requires explicit run_idx on every chain")
        return recs, namespace

    prod_recs, prod_namespace = select(prod_path, prod_construction)
    ref_recs, ref_namespace = select(ref_path, ref_construction)
    if prod_namespace != ref_namespace:
        raise ValueError("production and reference model/campaign must match")

    def index(recs):
        by = {}
        for r in recs:
            if r.get("error"):
                continue
            by.setdefault((r.get("k", 0), r["problem_id"]), {})[r["run_idx"]] = r
        return by

    prod_by, ref_by = index(prod_recs), index(ref_recs)
    common = sorted(set(prod_by) & set(ref_by))
    unpaired = sorted(set(prod_by) ^ set(ref_by))
    per_k: dict = {}
    insufficient = []
    unshared_runs = 0
    for (k, pid) in common:
        prod_runs, ref_runs = prod_by[(k, pid)], ref_by[(k, pid)]
        shared = sorted(set(prod_runs) & set(ref_runs))
        unshared_runs += len(set(prod_runs) ^ set(ref_runs))
        if len(shared) < 2:
            insufficient.append({"k": k, "problem_id": pid,
                                 "n_shared_runs": len(shared)})
            continue
        prod = [prod_runs[r] for r in shared]
        ref = [ref_runs[r] for r in shared]
        dp = determinism_max_vote([r["answer"] for r in prod])
        dr = determinism_max_vote([r["answer"] for r in ref])
        ap = mean(1.0 if r.get("correct") else 0.0 for r in prod)
        ar = mean(1.0 if r.get("correct") else 0.0 for r in ref)
        per_k.setdefault(k, []).append((dr - dp, ar - ap, len(shared)))
    rows = []
    flagged = False
    for k, gaps in sorted(per_k.items()):
        d_gap = mean(g[0] for g in gaps)
        acc_gap = mean(g[1] for g in gaps)
        rows.append({"k": k, "n_paired_problems": len(gaps),
                     "n_paired_runs": sum(g[2] for g in gaps),
                     "D_ref_minus_prod": round(d_gap, 4),
                     "acc_ref_minus_prod": round(acc_gap, 4)})
        if abs(d_gap) >= gap_threshold or abs(acc_gap) >= gap_threshold:
            flagged = True
    return {"rows": rows, "gap_threshold": gap_threshold,
            "n_unpaired_cells": len(unpaired),
            "n_unshared_runs": unshared_runs,
            "insufficient_shared_runs": insufficient,
            "construction_suspect": flagged,
            "passed": bool(rows) and not flagged and not unpaired and not insufficient}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("check-prompts")
    p1.add_argument("jsonl", type=Path)
    p1.add_argument("--problems", type=Path)
    p1.add_argument("--construction")
    p2 = sub.add_parser("gate-output")
    p2.add_argument("--file", type=Path)
    p3 = sub.add_parser("audit-log")
    p3.add_argument("jsonl", type=Path)
    p4 = sub.add_parser("calibrate")
    p4.add_argument("production_jsonl", type=Path)
    p4.add_argument("reference_jsonl", type=Path)
    p4.add_argument("--gap-threshold", type=float, default=0.15)
    p4.add_argument("--production-construction")
    p4.add_argument("--reference-construction")
    args = ap.parse_args()
    if args.cmd == "check-prompts":
        records = load_prompt_records(args.jsonl, args.problems,
                                      args.construction)
        result = check_prompt_records(records)
    elif args.cmd == "gate-output":
        text = (args.file.read_text() if args.file else sys.stdin.read())
        result = refusal_gate(text)
        result["passed"] = result["safe_to_parse"]
    elif args.cmd == "audit-log":
        result = audit_log(args.jsonl)
    else:
        result = calibrate(args.production_jsonl, args.reference_jsonl,
                           args.gap_threshold, args.production_construction,
                           args.reference_construction)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    sys.exit(main())
