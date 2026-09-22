"""Rebuild the local paper candidate and portable handoff; never submit it."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    out = ROOT / "output"
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "scripts/audit_evidence.py",
        "--summary",
        "artifact/results/summary.json",
    )
    run(sys.executable, "scripts/generate_paper.py")
    run(sys.executable, "scripts/make_figure.py")
    run(
        "tectonic",
        "-X",
        "compile",
        "main.tex",
        "--outdir",
        "../output/pdf",
        "--keep-logs",
        "--keep-intermediates",
        cwd=ROOT / "paper",
    )
    final = out / "pdf/aims2026-camera-ready.pdf"
    shutil.copyfile(out / "pdf/main.pdf", final)
    run("pdftotext", "-layout", str(final), str(out / "pdf/main.txt"))
    run(sys.executable, "scripts/check_paper.py")

    tex = (ROOT / "paper/main.tex").read_text()
    abstract = tex.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    macros = dict(
        re.findall(
            r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}",
            (ROOT / "paper/tables/numbers.tex").read_text(),
        )
    )
    for key, value in macros.items():
        abstract = abstract.replace("\\" + key + "{}", value)
    abstract = " ".join(abstract.split())
    metadata = {
        "title": "Phantom Measurement Artifacts in Multi-Step LLM Evaluation",
        "authors": ["Krishna Chaitanya Balusu"],
        "affiliation": "Independent Researcher",
        "affiliation_confirmation": "CONFIRMED",
        "abstract": abstract,
        "forum": "https://openreview.net/forum?id=966lGNYhCX",
    }
    (out / "openreview-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "OPENREVIEW_METADATA.md").write_text(
        "# OpenReview replacement metadata\n\n"
        "Title: " + metadata["title"] + "\n\n"
        "Author: Krishna Chaitanya Balusu\n\n"
        "Affiliation: Independent Researcher\n\n"
        "## Abstract\n\n" + abstract + "\n"
    )
    bundle = out / "aims2026-source-and-artifact.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for folder in ["paper", "artifact", "scripts", "evidence"]:
            for path in sorted((ROOT / folder).rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"
                    and path.relative_to(ROOT) != Path("paper/main.bbl")
                ):
                    archive.write(path, path.relative_to(ROOT))
        for name in ["claims.json", "requirements-build.txt"]:
            archive.write(ROOT / name, name)
        archive.write(out / "pdf/main.bbl", "paper/main.bbl")
        archive.writestr(
            "README.md",
            "# AIMS paper source and artifact\n\n"
            "Run `sh artifact/repro.sh` for offline reproduction with standard library Python.\n"
            "Install Tectonic and Poppler plus `pip install -r requirements-build.txt` to rebuild the paper with `python3 scripts/build.py`.\n"
            "No inference collection is invoked. See artifact/README.md for methods and provenance.\n",
        )
    delivered = [final, bundle, out / "openreview-metadata.json"]
    (out / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(path.read_bytes()).hexdigest()
            + "  "
            + str(path.relative_to(out))
            + "\n"
            for path in delivered
        )
    )
    print(
        "Prepared local PDF, source/artifact ZIP, and OpenReview metadata. No upload performed."
    )


if __name__ == "__main__":
    main()
