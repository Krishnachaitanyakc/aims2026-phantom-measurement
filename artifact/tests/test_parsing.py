import unittest

from harness.parsing import (answers_match, extract_answer, gold_from_gsm8k,
                             is_refusal, parse_output)


class TestExtract(unittest.TestCase):
    def test_hash_basic(self):
        self.assertEqual(extract_answer("blah\n#### 42"), ("42", "hash"))

    def test_hash_takes_last(self):
        self.assertEqual(extract_answer("#### 1\ntext\n#### 7"), ("7", "hash"))

    def test_hash_with_commas_and_dollar(self):
        self.assertEqual(extract_answer("#### $1,234.50"), ("1234.50", "hash"))

    def test_hash_negative(self):
        self.assertEqual(extract_answer("#### -3"), ("-3", "hash"))

    def test_fallback_last_numeric(self):
        ans, path = extract_answer("First 12 eggs then 30 dollars total")
        self.assertEqual((ans, path), ("30", "fallback"))

    def test_fallback_strips_percent(self):
        ans, path = extract_answer("the rate is 25%")
        self.assertEqual((ans, path), ("25", "fallback"))

    def test_none(self):
        self.assertEqual(extract_answer("I cannot answer."), ("", "none"))
        self.assertEqual(extract_answer(""), ("", "none"))


class TestMatch(unittest.TestCase):
    def test_numeric_equivalence(self):
        self.assertTrue(answers_match("1,000", "1000.0"))
        self.assertTrue(answers_match("42", "42"))
        self.assertTrue(answers_match("$5", "5"))

    def test_numeric_mismatch(self):
        self.assertFalse(answers_match("41", "42"))

    def test_empty_never_matches(self):
        self.assertFalse(answers_match("", ""))
        self.assertFalse(answers_match("", "5"))

    def test_string_fallback(self):
        self.assertTrue(answers_match("abc", "abc"))
        self.assertFalse(answers_match("abc", "abd"))


class TestRefusal(unittest.TestCase):
    def test_refusal_positive(self):
        self.assertTrue(is_refusal(
            "I don't see the problem statement. I need: 1. The problem"))
        self.assertTrue(is_refusal(
            "I don't see the actual calculation context or previous steps"))

    def test_refusal_negative(self):
        self.assertFalse(is_refusal("The answer is #### 42"))

    def test_parse_output_refusal_still_parses(self):
        out = parse_output("I don't see the problem statement. Section 3 says 12.")
        self.assertTrue(out["refusal"])
        self.assertEqual(out["answer"], "12")
        self.assertEqual(out["parse_path"], "fallback")


class TestGold(unittest.TestCase):
    def test_gold_extraction(self):
        self.assertEqual(gold_from_gsm8k(
            "Step <<3+4=7>> then <<7*2=14>>\n#### 14"), "14")

    def test_gold_with_commas(self):
        self.assertEqual(gold_from_gsm8k("work\n#### 1,200"), "1200")

    def test_gold_missing_raises(self):
        with self.assertRaises(ValueError):
            gold_from_gsm8k("no marker here")


if __name__ == "__main__":
    unittest.main()
