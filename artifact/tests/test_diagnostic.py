import json
import tempfile
import unittest
from pathlib import Path

from harness.diagnostic import (audit_log, calibrate, check_prompt_records,
                                load_prompt_records, problem_persists,
                                refusal_gate)


def rec(cons, k, pid, run_idx, answer, correct):
    return {"record_type": "chain", "construction": cons, "k": k,
            "problem_id": pid, "run_idx": run_idx, "answer": answer,
            "correct": correct, "error": None}


def write_jsonl(records):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


class TestCheckPrompts(unittest.TestCase):
    def test_pass_and_fail(self):
        good = [{"chain_id": 1, "step": s,
                 "prompt": f"step {s}: solve. Problem: what is 2+2?",
                 "problem": "What is 2+2?"} for s in range(3)]
        self.assertTrue(check_prompt_records(good)["passed"])
        bad = good + [{"chain_id": 2, "step": 9,
                       "prompt": "continue from before",
                       "problem": "What is 2+2?"}]
        res = check_prompt_records(bad)
        self.assertFalse(res["passed"])
        self.assertEqual(res["n_failures"], 1)

    def test_persistence_is_whitespace_case_insensitive(self):
        self.assertTrue(problem_persists("PROBLEM:  what   is 2+2?",
                                         "What is\n2+2?"))

    def test_empty_task_or_input_cannot_pass(self):
        self.assertFalse(problem_persists("arbitrary text", "  "))
        self.assertFalse(check_prompt_records([])["passed"])

    def test_native_adapter_uses_external_problem_and_latest_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory) / "panel.json"
            panel.write_text(json.dumps({"problems": [
                {"id": "p1", "question": "What is 2+2?"}]}))
            old = rec("fixed", 1, "p1", 0, "4", True)
            old["steps"] = [{"prompt": "unrelated"}]
            latest = dict(old, steps=[{"prompt": "What is 2+2?"}])
            degen = rec("degenerate", 1, "p1", 0, "4", True)
            degen["steps"] = [{"prompt": "Continue reasoning"}]
            path = write_jsonl([{"record_type": "meta"}, old, latest, degen])
            fixed = load_prompt_records(path, panel, "fixed")
            self.assertEqual(len(fixed), 1)
            self.assertTrue(check_prompt_records(fixed)["passed"])
            self.assertFalse(check_prompt_records(
                load_prompt_records(path, panel, "degenerate"))["passed"])
            with self.assertRaisesRegex(ValueError, "require --problems"):
                load_prompt_records(path)


class TestGate(unittest.TestCase):
    def test_gate(self):
        self.assertFalse(refusal_gate("I don't see the problem statement. "
                                      "Please provide it.")["safe_to_parse"])
        self.assertTrue(refusal_gate("#### 42")["safe_to_parse"])


class TestAuditLog(unittest.TestCase):
    def test_concentration_signature_flagged(self):
        recs = []
        for pid in ("p1", "p2"):
            for r in range(5):
                # k=5: scattered, one correct (D=0.2, acc=0.2)
                recs.append(rec("degen", 5, pid, r, "10" if r == 0 else str(r),
                                r == 0))
                # k=7: converged on a shared wrong answer (D=1.0, acc=0.0)
                recs.append(rec("degen", 7, pid, r, "99", False))
        res = audit_log(write_jsonl(recs))
        self.assertFalse(res["passed"])
        self.assertIn("FAILURE-MODE-CONCENTRATION", res["warnings"][0])

    def test_clean_log_passes(self):
        recs = [rec("fixed", k, p, r, "10", True)
                for k in (5, 7) for p in ("p1", "p2") for r in range(5)]
        self.assertTrue(audit_log(write_jsonl(recs))["passed"])


class TestCalibrate(unittest.TestCase):
    def test_no_shared_run_indices_cannot_pass_as_paired(self):
        prod = [rec("a", 7, "p1", r, "10", True) for r in (0, 1)]
        ref = [rec("b", 7, "p1", r, "10", True) for r in (2, 3)]
        res = calibrate(write_jsonl(prod), write_jsonl(ref))
        self.assertFalse(res["passed"])
        self.assertEqual(res["rows"], [])
        self.assertEqual(res["insufficient_shared_runs"][0]["n_shared_runs"], 0)

    def test_only_shared_runs_contribute_to_agreement_and_accuracy(self):
        prod = [rec("a", 7, "p1", r, "10", True) for r in (0, 1)]
        prod.append(rec("a", 7, "p1", 2, "99", False))
        ref = [rec("b", 7, "p1", r, "10", True) for r in (0, 1, 3)]
        res = calibrate(write_jsonl(prod), write_jsonl(ref))
        self.assertEqual(res["rows"][0]["D_ref_minus_prod"], 0)
        self.assertEqual(res["rows"][0]["acc_ref_minus_prod"], 0)
        self.assertEqual(res["rows"][0]["n_paired_runs"], 2)
        self.assertEqual(res["n_unshared_runs"], 2)

    def test_mixed_constructions_require_explicit_selection(self):
        rows = [rec(c, 7, "p1", r, "10", True)
                for c in ("fixed", "degenerate") for r in range(2)]
        path = write_jsonl(rows)
        with self.assertRaisesRegex(ValueError, "mixed constructions"):
            calibrate(path, path)
        res = calibrate(path, path, prod_construction="degenerate",
                        ref_construction="fixed")
        self.assertTrue(res["passed"])

    def test_mixed_campaigns_are_not_silently_deduplicated(self):
        rows = [dict(rec("fixed", 7, "p1", r, "10", True), campaign=c)
                for c in ("wave1", "wave2") for r in range(2)]
        with self.assertRaisesRegex(ValueError, "namespace"):
            audit_log(write_jsonl(rows))

    def test_different_model_namespaces_cannot_be_paired(self):
        prod = [dict(rec("a", 7, "p1", r, "10", True), arm_group="model-a")
                for r in range(2)]
        ref = [dict(rec("b", 7, "p1", r, "10", True), arm_group="model-b")
               for r in range(2)]
        with self.assertRaisesRegex(ValueError, "must match"):
            calibrate(write_jsonl(prod), write_jsonl(ref))

    def test_missing_run_indices_are_rejected(self):
        rows = [rec("a", 7, "p1", r, "10", True) for r in range(2)]
        for row in rows:
            del row["run_idx"]
        with self.assertRaisesRegex(ValueError, "explicit run_idx"):
            calibrate(write_jsonl(rows), write_jsonl(rows))

    def test_large_gap_flags_construction(self):
        prod = [rec("degen", 7, p, r, str(r), False)
                for p in ("p1", "p2") for r in range(5)]
        ref = [rec("fixed", 7, p, r, "10", True)
               for p in ("p1", "p2") for r in range(5)]
        res = calibrate(write_jsonl(prod), write_jsonl(ref))
        self.assertTrue(res["construction_suspect"])
        self.assertFalse(res["passed"])

    def test_matched_constructions_pass(self):
        prod = [rec("a", 7, p, r, "10", True)
                for p in ("p1", "p2") for r in range(5)]
        ref = [rec("b", 7, p, r, "10", True)
               for p in ("p1", "p2") for r in range(5)]
        self.assertTrue(calibrate(write_jsonl(prod),
                                  write_jsonl(ref))["passed"])


if __name__ == "__main__":
    unittest.main()
