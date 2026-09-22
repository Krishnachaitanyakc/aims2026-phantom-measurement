"""Mechanical PDF and source checks; release remains gated on open claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
HEADER = "Published at the AI Measurement Science Workshop at COLM 2026."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-release", action="store_true")
    args = parser.parse_args()
    pdf = ROOT / "output/pdf/aims2026-camera-ready.pdf"
    reader = PdfReader(pdf)
    texts = [page.extract_text() for page in reader.pages]
    assert len(texts) <= 9, "Conservative total-PDF page gate exceeded"
    for number, (page, text) in enumerate(zip(reader.pages, texts), 1):
        assert tuple(float(v) for v in page.mediabox[2:]) == (612.0, 792.0)
        assert HEADER in " ".join(text.split()), (
            f"Workshop header missing on page {number}"
        )
        assert "Under review as" not in text and "??" not in text
    source = (ROOT / "paper/main.tex").read_text()
    assert r"\usepackage[final]{colm2026_conference}" in source
    assert r"\linenumbers" not in source and "Anonymous" not in source
    assert not re.search(r"\b(TODO|FIXME|XXX)\b", source)
    assert "14--42" not in source and "14–42" not in source
    assert r"\geometry" not in source and r"\setlength{\text" not in source
    official = json.loads((ROOT / "evidence/template_hashes.json").read_text())
    for name, digest in official.items():
        assert (
            hashlib.sha256((ROOT / "paper" / name).read_bytes()).hexdigest() == digest
        )
    log = (ROOT / "output/pdf/main.log").read_text()
    for message in ["Overfull", "undefined references", "Citation ", "LaTeX Error"]:
        assert message not in log, message
    font_report = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    for line in font_report.splitlines()[2:]:
        fields = line.split()
        assert fields[-5] == "yes", f"Font not embedded: {line}"
        assert "Type 3" not in line
    uris = sorted(
        {
            str(a.get_object().get("/A", {}).get("/URI"))
            for p in reader.pages
            for a in p.get("/Annots", [])
            if a.get_object().get("/A", {}).get("/URI")
        }
    )
    ledger = json.loads((ROOT / "claims.json").read_text())
    open_claims = [r["id"] for r in ledger if r["status"] != "DISCHARGED"]
    for claim in ledger:
        if claim["kind"] == "artifact" and claim["status"] == "DISCHARGED":
            target = claim["discharged_by"]
            assert target.startswith("url:") and target[4:] in uris, (
                "Discharged artifact URL is absent from the PDF annotations"
            )
    report = {
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "total_pages": len(texts),
        "references_start_page": next(
            i for i, text in enumerate(texts, 1) if "\nReferences" in text
        ),
        "paper_size": "US Letter",
        "all_fonts_embedded": True,
        "review_line_numbers": False,
        "official_template_unchanged": True,
        "header_correct_on_every_page": True,
        "all_local_pdf_checks": "PASS",
        "open_claims": open_claims,
        "uri_annotations": uris,
        "submission_readiness": "NOT_DECLARED"
        if open_claims
        else "TECHNICAL_CHECKS_PASS",
    }
    (ROOT / "evidence/pdf_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if args.require_release and open_claims:
        raise SystemExit(
            "Release blocked: unresolved claim(s) " + ", ".join(open_claims)
        )


if __name__ == "__main__":
    main()
