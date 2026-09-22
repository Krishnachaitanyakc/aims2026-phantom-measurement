"""GSM8K acquisition and deterministic stratified subset selection.

Source of record: openai/grade-school-math test split (MIT license), fetched
directly from the upstream repository and SHA-256 pinned. The historical
50-problem subset is unrecoverable (its loader was lost); this campaign defines
a NEW documented subset and publishes the exact problem IDs.
"""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

from .parsing import gold_from_gsm8k

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
EXPECTED_N = 1319
EXPECTED_SHA256 = ("3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4"
                   "b3c39d14")

# Stratification: number of `<<...>>` calculator annotations in the gold
# solution (a step-count proxy). Thresholds pre-registered in EXPERIMENT_PLAN.md.
EASY_MAX = 2      # n_steps <= 2
HARD_MIN = 6      # n_steps >= 6
STRATA_SIZES = {"easy": 3, "medium": 6, "hard": 3}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def download_gsm8k(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try:
            req = urllib.request.Request(
                GSM8K_TEST_URL, headers={"User-Agent": "coling2027-artifact/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
        except Exception:
            # Stock macOS framework Pythons often lack CA certs; curl uses the
            # system trust store.
            import subprocess
            subprocess.run(["curl", "-fsSL", "-o", str(dest), GSM8K_TEST_URL],
                           check=True, timeout=120)
    n = sum(1 for line in dest.read_text().splitlines() if line.strip())
    if n != EXPECTED_N:
        raise RuntimeError(f"GSM8K test split has {n} lines; expected {EXPECTED_N}")
    got = sha256_file(dest)
    if got != EXPECTED_SHA256:
        raise RuntimeError(
            f"GSM8K test split SHA-256 mismatch: got {got}, "
            f"expected {EXPECTED_SHA256} — refusing to proceed")
    return dest


def load_problems(path: Path) -> list[dict]:
    problems = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        n_steps = rec["answer"].count("<<")
        stratum = ("easy" if n_steps <= EASY_MAX
                   else "hard" if n_steps >= HARD_MIN else "medium")
        problems.append({
            "id": f"gsm8k_test_{i}",
            "index": i,
            "question": rec["question"].strip(),
            "gold": gold_from_gsm8k(rec["answer"]),
            "n_steps": n_steps,
            "stratum": stratum,
        })
    return problems


def select_subset(problems: list[dict], seed: int,
                  sizes: dict[str, int] | None = None) -> list[dict]:
    sizes = sizes or STRATA_SIZES
    rng = random.Random(seed)
    chosen: list[dict] = []
    for stratum in ("easy", "medium", "hard"):
        pool = [p for p in problems if p["stratum"] == stratum]
        if len(pool) < sizes[stratum]:
            raise RuntimeError(f"stratum {stratum} too small: {len(pool)}")
        chosen.extend(rng.sample(sorted(pool, key=lambda p: p["index"]),
                                 sizes[stratum]))
    return sorted(chosen, key=lambda p: p["index"])


def build_subset_file(data_dir: Path, seed: int) -> Path:
    src = download_gsm8k(data_dir / "gsm8k_test.jsonl")
    problems = load_problems(src)
    subset = select_subset(problems, seed)
    out = data_dir / f"subset_seed{seed}.json"
    out.write_text(json.dumps({
        "seed": seed,
        "source_url": GSM8K_TEST_URL,
        "source_sha256": sha256_file(src),
        "strata_def": {"easy_max_steps": EASY_MAX, "hard_min_steps": HARD_MIN,
                       "sizes": STRATA_SIZES},
        "ids": [p["id"] for p in subset],
        "problems": subset,
    }, indent=2))
    return out


def load_subset(data_dir: Path, seed: int) -> list[dict]:
    path = data_dir / f"subset_seed{seed}.json"
    if not path.exists():
        path = build_subset_file(data_dir, seed)
    return json.loads(path.read_text())["problems"]


def build_addendum_file(data_dir: Path, seed: int, base_seed: int) -> Path:
    """Select a disjoint additional subset (Addendum R2)."""
    src = download_gsm8k(data_dir / "gsm8k_test.jsonl")
    base_ids = {p["id"] for p in load_subset(data_dir, base_seed)}
    problems = [p for p in load_problems(src) if p["id"] not in base_ids]
    subset = select_subset(problems, seed)
    out = data_dir / f"subset_seed{seed}_addendum.json"
    out.write_text(json.dumps({
        "seed": seed, "disjoint_from_seed": base_seed,
        "source_url": GSM8K_TEST_URL, "source_sha256": sha256_file(src),
        "strata_def": {"easy_max_steps": EASY_MAX, "hard_min_steps": HARD_MIN,
                       "sizes": STRATA_SIZES},
        "ids": [p["id"] for p in subset], "problems": subset,
    }, indent=2))
    return out


def load_addendum(data_dir: Path, seed: int, base_seed: int) -> list[dict]:
    path = data_dir / f"subset_seed{seed}_addendum.json"
    if not path.exists():
        path = build_addendum_file(data_dir, seed, base_seed)
    return json.loads(path.read_text())["problems"]
