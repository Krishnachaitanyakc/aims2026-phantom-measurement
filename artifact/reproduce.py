"""Recompute only the distributed R3 evidence; no API calls or downloads."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from harness.analyze import analyze, load_chains, write_refusal_examples
from verify_inputs import ROOT, verify_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    receipt = verify_inputs()
    loaded = load_chains(ROOT / "runs")
    if len(loaded["latest"]) != 1260:
        raise ValueError("Expected 1,260 selected R3 chains")
    if any(r["_class"] != "conf" for r in loaded["latest"]):
        raise ValueError("Only the R3 campaign is in scope")
    groups = {r["_group"] for r in loaded["latest"]}
    if groups != {"haiku", "sonnet", "codex-gpt56"}:
        raise ValueError("Incomplete model family")
    summary = analyze(loaded)
    for group in groups:
        if set(summary["arms"][group]) != {"conf"}:
            raise ValueError("Unexpected campaign class in the summary")
    out = args.out_dir or Path(tempfile.mkdtemp(prefix="aims2026-reproduction-"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "input_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    write_refusal_examples(loaded["latest"], out)
    print(f"Reproduced 1,260 selected chains from 1,342 stored chain records: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
