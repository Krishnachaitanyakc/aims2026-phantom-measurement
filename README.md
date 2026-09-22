# Phantom Measurement Artifacts

Reproduction artifact and source for the accepted AIMS 2026 paper, “Phantom Measurement Artifacts in Multi-Step LLM Evaluation,” by Krishna Chaitanya Balusu, Independent Researcher.

The retained logs cover 1,260 selected chains and 5,940 calls on a fixed panel of 12 GSM8K problems. The analysis examines question omission, numeric answer extraction, and final call restoration. The results concern this panel and the recorded hosted interfaces.

## Reproduce

Python 3.10 or later is sufficient. Run from the repository root:

```sh
sh artifact/repro.sh
```

The command checks input hashes, runs 85 tests, and rebuilds the results in a temporary directory. It makes no model calls. See [artifact/README.md](artifact/README.md) for the design, selection rule, prompts, and provenance.

## Paper and source

The [versioned release](https://github.com/Krishnachaitanyakc/aims2026-phantom-measurement/releases/tag/v1.0.0) contains the PDF, source and artifact ZIP, and SHA256 checksums. Install Tectonic, Poppler, and the Python build dependencies to rebuild the paper:

```sh
python3 -m pip install -r requirements-build.txt
python3 scripts/build.py
```

The build audits the raw evidence, regenerates tables and the figure, compiles the paper, and checks the PDF. The workshop is nonarchival. This repository release does not establish a final OpenReview upload.

## Rights

Author code retains its [MIT license](artifact/LICENSE). GSM8K retains [OpenAI's MIT notice](artifact/licenses/GSM8K-MIT.txt). Model outputs are not relicensed by those notices; see [license provenance](artifact/licenses/README.md). The official COLM template files retain their original notices.
