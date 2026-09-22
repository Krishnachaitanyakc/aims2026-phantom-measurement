#!/usr/bin/env python3
"""Generate AIMS tables from independently audited integer counts and contrasts."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("haiku", "Haiku 4.5"),
    ("codex-gpt56", "GPT 5.6 Sol"),
    ("sonnet", "Sonnet 5"),
]


def table(caption, label, columns, header, rows):
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{@{}" + columns + "@{}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "evidence/camera_ready_evidence.json"
    )
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text())
    out = ROOT / "paper/tables"
    out.mkdir(exist_ok=True, parents=True)
    models = data["models"]
    h = models["haiku"]["cells"]
    gaps = [
        100
        * float(
            Fraction(
                models[k]["contrasts"]["fixed_minus_degenerate_k7"]["D"]["exact_gap"]
            )
        )
        for k, _ in MODELS
    ]
    fixed_correct = [models[k]["cells"]["fixed_k7"]["correct_count"] for k, _ in MODELS]
    omitted_correct = [
        models[k]["cells"]["degenerate_k7"]["correct_count"] for k, _ in MODELS
    ]
    numbers = {
        "GapRange": f"{min(gaps):.1f} to {max(gaps):.1f}",
        "HaikuDfive": f"{h['degenerate_k5']['D']:.3f}",
        "HaikuDseven": f"{h['degenerate_k7']['D']:.3f}",
        "HaikuAccFiveCount": h["degenerate_k5"]["correct_count"],
        "HaikuAccSevenCount": h["degenerate_k7"]["correct_count"],
        "FallbackCount": h["degenerate_k7"]["parse_counts"]["fallback"],
        "StepCount": h["degenerate_k7"]["fallback_values"]["6"],
        "HaikuStrictD": f"{h['degenerate_k7']['D_strict']:.3f}",
        "FixedAccuracyRange": f"{min(fixed_correct)} to {max(fixed_correct)}",
        "OmittedAccuracyRange": f"{min(omitted_correct)} to {max(omitted_correct)}",
    }
    (out / "numbers.tex").write_text(
        "% Generated; do not edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numbers.items())
        + "\n"
    )

    rows = []
    for key, name in MODELS:
        x = models[key]["contrasts"]["fixed_minus_degenerate_k7"]
        for idx, (metric, label, ci_key) in enumerate(
            [
                ("D", r"$D$", "D_gap_ci"),
                ("D_strict", r"$D_{\rm strict}$", "D_gap_strict_ci"),
                ("acc", "Accuracy", "acc_gap_ci"),
                ("acc_strict", "Strict accuracy", "acc_strict_gap_ci"),
            ]
        ):
            gap = float(Fraction(x[metric]["exact_gap"])) * 100
            lo, hi = (100 * v for v in x["reproduced_intervals"][ci_key])
            p = x[metric]["holm_p_across_three_models"]
            rows.append(
                f"{name if idx == 0 else ''} & {label} & {gap:.1f} [{lo:.1f}, {hi:.1f}] & {p:.4f}"
                + r" \\"
            )
        if key != MODELS[-1][0]:
            rows.append(r"\addlinespace")
    (out / "contrasts.tex").write_text(
        table(
            r"Question restoration increases agreement and accuracy at depth seven. Gaps are \fixed{} minus \dropq{}, in percentage points, on 12 paired problems with five chains per condition. Intervals are problem bootstrap 95\% BCa intervals, not simultaneous coverage intervals. Holm $p$ adjusts the exact paired sign flip test across three models within each metric; inference assumes sign symmetry. Strict scores are sensitivity endpoints.",
            "tab:contrasts",
            "llrr",
            r"Model & Metric & Gap [95\% CI], pp & Holm $p$",
            rows,
        )
    )

    rows = []
    for key, name in MODELS:
        cells = models[key]["cells"]
        for i, (cell_key, label, depth) in enumerate(
            [
                ("degenerate_k1", "Both at $k=1$", 1),
                ("degenerate_k5", r"\dropq", 5),
                ("degenerate_k7", r"\dropq", 7),
                ("fixed_k5", r"\fixed", 5),
                ("fixed_k7", r"\fixed", 7),
                ("finalonly_k7", r"\finalonly", 7),
            ]
        ):
            c = cells[cell_key]
            assert (
                c["n_chains"] == 60
                and c["n_problems"] == 12
                and c["n_runs_per_problem"] == 5
            )
            assert abs(c["D"] - c["modal_votes_total"] / 60) < 0.000051
            assert abs(c["acc"] - c["correct_count"] / 60) < 0.000051
            if depth == 1:
                assert (
                    c["D"] == cells["fixed_k1"]["D"] == 1
                    and c["correct_count"] == cells["fixed_k1"]["correct_count"] == 60
                )
            rows.append(
                f"{name if i == 0 else ''} & {label} & {depth} & {c['D']:.3f} & {c['D_strict']:.3f} & {c['correct_count']}/60 & {c['strict_correct_count']}/60"
                + r" \\"
            )
        if key != MODELS[-1][0]:
            rows.append(r"\addlinespace")
    (out / "cells.tex").write_text(
        table(
            r"Final call restoration recovers correct answers on this panel. Each row contains 12 problems $\times$ five chains. The two $k=1$ cells per model are displayed together because both have identical observed values; each still contains 60 separate chains. $D$ and $D_{\rm strict}$ are mean per problem plurality shares. Accuracy columns give correct final answers, with rejected runs retained as incorrect. Ceiling scores and final control ties are descriptive, not equivalence evidence.",
            "tab:cells",
            "llrrrrr",
            r"Model & Construction & $k$ & $D$ & $D_{\rm strict}$ & Correct & Strict",
            rows,
        )
    )

    rows = []
    for depth in [5, 7]:
        c = h[f"degenerate_k{depth}"]
        step_count = c["fallback_values"].get(str(depth - 1), 0)
        rows.append(
            f"{depth} & {c['parse_counts'].get('hash', 0)} & {c['parse_counts'].get('fallback', 0)} & {step_count}/{c['parse_counts'].get('fallback', 0)} ({depth - 1}) & {c['D']:.3f} & {c['D_marker_only']:.3f} & {c['D_strict']:.3f}"
            + r" \\"
        )
    (out / "parser.tex").write_text(
        table(
            r"Haiku agreement rebounds only with permissive scoring. Each row has 60 final outputs under \dropq. Hash and fallback extraction paths partition those outputs. The step column reports fallback values equal to the immediately preceding step index (shown in parentheses), over all fallback extractions. Marker only removes fallback extraction without using the phrase gate; it is a post hoc descriptive sensitivity. Literal token counts are not human semantic labels. Correctness is reported in Table~\ref{tab:cells}.",
            "tab:parser",
            "rrrrrrr",
            r"$k$ & Hash & Fallback & Step token & $D$ & Marker only & Strict",
            rows,
        )
    )

    rows = []
    for key, name in MODELS:
        m = models[key]
        counts = m["counts"]
        model_id = m["observed_model_ids"][0].replace("-", r"\allowbreak-")
        dates = (
            "July 30 to 31"
            if m["selected_last_call_utc"][:10] != m["selected_first_call_utc"][:10]
            else "July 30"
        )
        rows.append(
            r"\multicolumn{4}{@{}l}{"
            + name
            + ": "
            + r"\texttt{"
            + model_id
            + "}}"
            + r" \\"
        )
        rows.append(
            f"{dates} & {counts['stored_chain_records']} & {counts['selected_step_calls']:,} & {counts['inference_attempts']:,}"
            + r" \\"
        )
        if key != MODELS[-1][0]:
            rows.append(r"\addlinespace")
    (out / "provenance.tex").write_text(
        table(
            r"Retained R3 collection, July 2026 (UTC). Each model contributes 420 analyzed chains and 1,980 selected calls. Stored records include failed or superseded chains; attempts include retries and superseded work. Latest complete records are selected by the analysis. No selected chain has a call error or observed identity deviation. CLI versions recorded in the protocol are Claude Code 2.1.220 and Codex 0.145.0; they are not independently attested in each call.",
            "tab:provenance",
            "lrrr",
            "Selected dates & Stored chains & Selected calls & All attempts",
            rows,
        )
    )
    print(
        "Generated five tables and eleven shared numerical macros from audited evidence."
    )


if __name__ == "__main__":
    main()
