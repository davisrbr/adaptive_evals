import unittest
from counter import count_words


class TestCountWords(unittest.TestCase):
    def test_simple(self):
        result = count_words("hello world hello")
        self.assertEqual(result["hello"], 2)
        self.assertEqual(result["world"], 1)

    def test_case_insensitive(self):
        result = count_words("Hello hello HELLO")
        self.assertEqual(result["hello"], 3)
        self.assertEqual(len(result), 1)

    def test_empty(self):
        self.assertEqual(count_words(""), {})


if __name__ == "__main__":
    unittest.main()
