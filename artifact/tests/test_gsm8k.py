import unittest

from harness.gsm8k import EASY_MAX, HARD_MIN, select_subset


def prob(i, n_steps):
    stratum = ("easy" if n_steps <= EASY_MAX
               else "hard" if n_steps >= HARD_MIN else "medium")
    return {"id": f"gsm8k_test_{i}", "index": i, "question": f"q{i}",
            "gold": str(i), "n_steps": n_steps, "stratum": stratum}


def pool():
    problems = []
    i = 0
    for n_steps, count in ((1, 30), (2, 30), (3, 40), (4, 40), (5, 40),
                           (6, 20), (7, 20)):
        for _ in range(count):
            problems.append(prob(i, n_steps))
            i += 1
    return problems


class TestSubset(unittest.TestCase):
    def test_sizes_and_strata(self):
        subset = select_subset(pool(), seed=20260728)
        self.assertEqual(len(subset), 12)
        counts = {}
        for p in subset:
            counts[p["stratum"]] = counts.get(p["stratum"], 0) + 1
        self.assertEqual(counts, {"easy": 3, "medium": 6, "hard": 3})

    def test_deterministic(self):
        a = [p["id"] for p in select_subset(pool(), seed=20260728)]
        b = [p["id"] for p in select_subset(pool(), seed=20260728)]
        self.assertEqual(a, b)

    def test_seed_sensitivity(self):
        a = [p["id"] for p in select_subset(pool(), seed=1)]
        b = [p["id"] for p in select_subset(pool(), seed=2)]
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
