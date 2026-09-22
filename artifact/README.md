# AIMS 2026 paper artifact

This bundle reproduces the GSM8K replication used in the revised paper. It
contains the `SW-JUL26-R3` logs for Claude Haiku 4.5, Claude Sonnet 5 and
GPT 5.6 sol: 12 problems, five chains per construction and depth, and
1,260 retained chains. No new inference is needed.

## Reproduce

Use Python 3.10 or later. Only the standard library is required.

```sh
sh repro.sh
```

The command checks input hashes, runs the tests and rebuilds the results.
It writes into a new temporary directory and prints that directory at the
end. To choose an output directory, run `sh repro.sh /tmp/aims-results`.
The included commands make no network requests and consume no model quota.
The independent raw evidence audit in the enclosing paper package is run
when `../scripts/audit_evidence.py` is present.

For analysis alone:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reproduce.py --out-dir /tmp/aims-results
```

## Evidence and methods

`data/` contains the complete GSM8K test split and the exact 12 problem
panel selected with seed `20260728`. The source hash is checked before
analysis. `runs/` contains three original campaign logs, retaining failed
or superseded attempts. Analysis selects the latest record for each
model, campaign, construction, depth, problem and run index.
The Haiku log contains 502 chain records, of which 82 are superseded:
75 failed records and seven complete records. A rerun of a collection unit
can replace successful records alongside failed ones. Selection follows
record order, not answer correctness or agreement. Superseded records
remain available for inspection.

The selected grid includes `fixed` and `degenerate` at depths 1, 5 and 7,
plus `finalonly` at depth 7. Each model has 420 retained chains. Logs store
rendered prompts, model outputs, parsing results, model identities,
timestamps and attempt counts. Two private home directory prefixes were
replaced by `<HOME>` in a Haiku output and the following prompt. That
prompt's hash was recomputed. `PROVENANCE.json` records the original and
distributed hashes and the locations of both replacements.
The old top level `campaign` value in native metadata records is retained
for provenance. Per chain `campaign` and metadata `arm_config.campaign`
identify the R3 wave and agree.

`harness/analyze.py` retains the campaign analysis routines so that the
paper estimates can be compared with the source analysis. The narrow
entry point accepts only the three distributed R3 logs. It does not
substitute earlier campaigns or generate estimates for absent waves.
The original table and figure generators are excluded.

The primary quantity is parsed answer agreement, the largest answer
frequency divided by five, averaged over problems. Accuracy is recorded
separately. Strict scoring assigns each refusal or answer without the
required `####` marker a distinct no answer token. Intervals use a
problem bootstrap. Sign flip p values require a sign symmetry or
exchangeability assumption. These are hosted CLI measurements with
uncontrolled vendor decoding defaults and coding agent scaffolds.

The results concern this small arithmetic panel. They do not establish
defect prevalence, general model rankings or reliable accumulation of
intermediate reasoning. Final step results are descriptive; the paper
does not claim equivalence from the retained TOST diagnostics.

## Inspect the prompts

```sh
python3 -m harness.diagnostic check-prompts runs/haiku-r3.jsonl \
  --problems data/subset_seed20260728.json --construction fixed
python3 -m harness.diagnostic check-prompts runs/haiku-r3.jsonl \
  --problems data/subset_seed20260728.json --construction degenerate
```

The second command returns status 2 because the omitted problem statement
is the injected defect. The adapter gets the original problem from the
panel and selects the latest native chain records before checking.

```sh
python3 -m harness.diagnostic calibrate runs/haiku-r3.jsonl \
  runs/haiku-r3.jsonl --production-construction degenerate \
  --reference-construction fixed
```

Calibration uses shared run indices within each problem and depth. It
rejects mixed model or campaign inputs and requires explicit selection
when a file contains several constructions. Cells with fewer than two
shared runs cannot pass. Status 2 means that the diagnostic found a large
gap or insufficient paired evidence. A passing check is not a guarantee
that an evaluation harness is correct.

## Provenance and rights

The source data, code and changes are recorded in `PROVENANCE.json`.
The author code retains its existing MIT license in `LICENSE`, with the
author name restored for the camera ready version.
The accompanying `licenses/GSM8K-MIT.txt` is the license supplied by
OpenAI for GSM8K. Its provenance is in `licenses/README.md`.
Model outputs remain subject to the respective vendors' terms and are
not relicensed by the author code's MIT grant. Release versions and checksums
are maintained in the enclosing repository.
