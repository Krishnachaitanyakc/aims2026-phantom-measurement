import unittest

from harness.diagnostic import problem_persists
from harness.prompts import (BUILDERS, build_chain_degenerate,
                             build_chain_fixed, build_chain_padded,
                             filler_text, raw_prompt, render_step)

Q = "Natalia sold clips to 48 of her friends in April, and then she sold " \
    "half as many clips in May. How many clips did Natalia sell altogether?"


class TestConstructions(unittest.TestCase):
    def test_fixed_persists_problem_every_step(self):
        for k in (2, 3, 5, 7):
            chain = build_chain_fixed(Q, k)
            self.assertEqual(len(chain), k)
            for step_prompt in chain:
                self.assertTrue(problem_persists(step_prompt, Q))

    def test_degenerate_loses_problem_after_step1(self):
        for k in (2, 3, 5, 7):
            chain = build_chain_degenerate(Q, k)
            self.assertEqual(len(chain), k)
            self.assertTrue(problem_persists(chain[0], Q))
            for step_prompt in chain[1:]:
                self.assertFalse(problem_persists(step_prompt, Q))

    def test_padded_matches_fixed_length_without_problem(self):
        for k in (3, 7):
            fixed = build_chain_fixed(Q, k)
            padded = build_chain_padded(Q, k)
            self.assertEqual(len(padded), k)
            self.assertEqual(padded[0], fixed[0])  # step 1 identical
            for f, p in zip(fixed[1:], padded[1:]):
                self.assertFalse(problem_persists(p, Q))
                # length within 15% of the fixed counterpart
                self.assertLess(abs(len(p) - len(f)) / len(f), 0.15)

    def test_filler_has_no_digits(self):
        self.assertFalse(any(ch.isdigit() for ch in filler_text(500)))

    def test_filler_exact_length(self):
        for n in (10, 137, 900):
            self.assertEqual(len(filler_text(n)), n)

    def test_final_step_requests_hash_format(self):
        for builder in BUILDERS.values():
            self.assertIn("#### ", builder(Q, 4)[-1])

    def test_raw_prompt_contains_problem_and_format(self):
        rp = raw_prompt(Q)
        self.assertTrue(problem_persists(rp, Q))
        self.assertIn("#### ", rp)

    def test_render_step(self):
        tmpl = "before\n{PREV_OUTPUT}\nafter"
        self.assertEqual(render_step(tmpl, "XYZ"), "before\nXYZ\nafter")

    def test_k2_edge(self):
        chain = build_chain_degenerate(Q, 2)
        self.assertEqual(len(chain), 2)
        self.assertIn("final answer", chain[1])


if __name__ == "__main__":
    unittest.main()


class TestR2Constructions(unittest.TestCase):
    def test_finalonly_problem_at_ends_only(self):
        from harness.prompts import build_chain_finalonly
        for k in (3, 7):
            chain = build_chain_finalonly(Q, k)
            self.assertEqual(len(chain), k)
            self.assertTrue(problem_persists(chain[0], Q))
            self.assertTrue(problem_persists(chain[-1], Q))
            for mid in chain[1:-1]:
                self.assertFalse(problem_persists(mid, Q))
            self.assertIn("#### ", chain[-1])

    def test_padded2_token_matched_no_problem(self):
        from harness.prompts import build_chain_fixed, build_chain_padded2
        for k in (3, 7):
            fixed = build_chain_fixed(Q, k)
            p2 = build_chain_padded2(Q, k)
            self.assertEqual(p2[0], fixed[0])
            for f, p in zip(fixed[1:], p2[1:]):
                self.assertFalse(problem_persists(p, Q))
                self.assertLessEqual(abs(len(p.split()) - len(f.split())), 1)

    def test_padded2_filler_digit_free(self):
        from harness.prompts import build_chain_padded2
        chain = build_chain_padded2(Q, 7)
        for prompt in chain[1:]:
            for line in prompt.splitlines():
                if line.startswith("Note (") or line.startswith("Keep your"):
                    self.assertFalse(any(c.isdigit() for c in line))
