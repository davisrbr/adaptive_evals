import unittest
from dataproc import merge_records


class TestMergeRecords(unittest.TestCase):
    def test_no_overlap(self):
        result = merge_records([{"a": 1}, {"b": 2}])
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_overlap(self):
        result = merge_records([{"a": 1}, {"a": 2}, {"b": 3}])
        self.assertEqual(result["a"], [1, 2])
        self.assertEqual(result["b"], 3)

    def test_empty(self):
        self.assertEqual(merge_records([]), {})


if __name__ == "__main__":
    unittest.main()
