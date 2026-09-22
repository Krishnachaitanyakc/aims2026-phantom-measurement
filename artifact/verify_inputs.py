"""Check the shipped evidence without network access or file writes."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from harness.gsm8k import EXPECTED_SHA256

ROOT = Path(__file__).resolve().parent
RUN_FILES = {"haiku-r3.jsonl", "codex-r3.jsonl", "sonnet-r3.jsonl"}


def verify_inputs() -> dict:
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    checked = []
    for record in provenance["copied_files"]:
        path = record["path"]
        if not path.startswith(("runs/", "data/")):
            continue
        payload = (ROOT / path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["initial_distribution_sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {path}")
        checked.append(path)
    actual_runs = {p.name for p in (ROOT / "runs").glob("*.jsonl")}
    if actual_runs != RUN_FILES:
        raise ValueError(f"Expected exactly the three R3 logs; found {sorted(actual_runs)}")
    data_sha = hashlib.sha256((ROOT / "data/gsm8k_test.jsonl").read_bytes()).hexdigest()
    if data_sha != EXPECTED_SHA256:
        raise ValueError("GSM8K source pin mismatch")
    panel = json.loads((ROOT / "data/subset_seed20260728.json").read_text())
    if panel["source_sha256"] != data_sha or len(set(panel["ids"])) != 12:
        raise ValueError("Invalid pinned panel")
    privacy_patterns = {
        "private-home-path": r"/(?:Users|home)/[^/\s\"\\]+",
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "credential-shape": r"(?i)(?:bearer\s+[a-z\d_.-]{15,}|sk-[a-z\d_-]{15,}|gh[pousr]_[a-z\d_]+)",
    }
    n_chain_records = 0
    for name in sorted(RUN_FILES):
        text = (ROOT / "runs" / name).read_text()
        for label, pattern in privacy_patterns.items():
            if re.search(pattern, text):
                raise ValueError(f"Privacy pattern {label} in {name}; inspect locally")
        for line in text.splitlines():
            record = json.loads(line)
            if record.get("record_type") != "chain":
                continue
            n_chain_records += 1
            if record["campaign"] != "SW-JUL26-R3":
                raise ValueError(f"Unexpected campaign in {name}")
            for step in record["steps"]:
                expected = hashlib.sha256(step["prompt"].encode()).hexdigest()
                if expected != step["prompt_sha256"]:
                    raise ValueError(f"Prompt hash mismatch in {name}")
    return {"checked_inputs": checked, "stored_chain_records": n_chain_records,
            "dataset_sha256": data_sha, "privacy_pattern_scan": "passed",
            "documented_redacted_fields": len(provenance["redactions"])}


if __name__ == "__main__":
    print(json.dumps(verify_inputs(), indent=2))
