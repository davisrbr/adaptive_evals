import unittest
from mathlib import summarize


class TestSummarize(unittest.TestCase):
    def test_nonempty(self):
        result = summarize([1, 2, 3])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["min"], 1)
        self.assertEqual(result["max"], 3)

    def test_empty(self):
        result = summarize([])
        self.assertEqual(result["mean"], 0.0)
        self.assertIsNone(result["min"])
        self.assertIsNone(result["max"])

    def test_single(self):
        result = summarize([42])
        self.assertEqual(result["mean"], 42.0)


if __name__ == "__main__":
    unittest.main()
