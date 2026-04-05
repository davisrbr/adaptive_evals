import unittest
from textproc import clean_text


class TestCleanText(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(clean_text("hello  world"), "hello world")

    def test_emoji(self):
        result = clean_text("hello \U0001f30d world")
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("\U0001f30d", result)

    def test_accents(self):
        result = clean_text("caf\u00e9 r\u00e9sum\u00e9")
        self.assertIn("caf\u00e9", result)
        self.assertIn("r\u00e9sum\u00e9", result)


if __name__ == "__main__":
    unittest.main()
