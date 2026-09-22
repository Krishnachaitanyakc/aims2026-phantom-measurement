import unittest
from statistics import mean

from harness.metrics import (bca_ci, determinism_max_vote, holm_correct,
                             paired_permutation_p, tara)


class TestD(unittest.TestCase):
    def test_unanimous(self):
        self.assertEqual(determinism_max_vote(["5", "5.0", "5"]), 1.0)

    def test_majority(self):
        self.assertEqual(determinism_max_vote(["1", "1", "1", "2", "2"]), 0.6)

    def test_uniform_dispersion(self):
        self.assertAlmostEqual(
            determinism_max_vote(["1", "2", "3", "4", "5"]), 0.2)

    def test_empty_answers_are_singletons(self):
        # two unparseable outputs must NOT count as agreeing
        self.assertEqual(determinism_max_vote(["", "", "7"]), 1 / 3)

    def test_tara(self):
        self.assertEqual(tara(["4", "4.0", "4"]), 1.0)
        self.assertEqual(tara(["4", "5", "4"]), 0.0)


class TestBootstrap(unittest.TestCase):
    def test_ci_contains_mean(self):
        vals = [0.2, 0.4, 0.6, 0.8, 1.0, 0.4, 0.6, 0.8, 1.0, 0.6, 0.4, 0.8]
        lo, hi = bca_ci(vals, b=2000, seed=1)
        self.assertLessEqual(lo, mean(vals))
        self.assertGreaterEqual(hi, mean(vals))
        self.assertLess(lo, hi)

    def test_degenerate_distribution(self):
        lo, hi = bca_ci([0.5] * 10, b=500, seed=1)
        self.assertEqual((lo, hi), (0.5, 0.5))

    def test_deterministic_given_seed(self):
        vals = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0]
        self.assertEqual(bca_ci(vals, b=1000, seed=7),
                         bca_ci(vals, b=1000, seed=7))


class TestPermutation(unittest.TestCase):
    def test_strong_effect_small_p(self):
        gaps = [0.3, 0.4, 0.35, 0.5, 0.3, 0.45, 0.4, 0.3, 0.35, 0.4, 0.5, 0.3]
        p = paired_permutation_p(gaps)
        self.assertLessEqual(p, 2 / 4096 + 1e-9)

    def test_null_effect_large_p(self):
        gaps = [0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.0, 0.0, 0.1, -0.1,
                -0.03, 0.03]
        self.assertGreater(paired_permutation_p(gaps), 0.5)

    def test_sign_symmetry(self):
        gaps = [0.2, 0.1, 0.3, -0.1, 0.25, 0.15]
        self.assertAlmostEqual(paired_permutation_p(gaps),
                               paired_permutation_p([-g for g in gaps]))


class TestHolm(unittest.TestCase):
    def test_ordering_and_monotonicity(self):
        adj = holm_correct({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertGreaterEqual(adj["b"], adj["c"])
        self.assertTrue(all(0 <= v <= 1 for v in adj.values()))


if __name__ == "__main__":
    unittest.main()
