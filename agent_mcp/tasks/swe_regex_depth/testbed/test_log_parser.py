import unittest
from log_parser import parse_log_levels


class TestParseLogLevels(unittest.TestCase):
    def test_uppercase(self):
        text = "[INFO] Server started\n[ERROR] Connection failed"
        result = parse_log_levels(text)
        self.assertIn("INFO", result)
        self.assertIn("ERROR", result)

    def test_case_insensitive(self):
        text = "[info] low message\n[Warning] mid message\n[ERROR] high message"
        result = parse_log_levels(text)
        # All levels should be normalized to uppercase
        self.assertEqual(len(result.get("INFO", [])), 1)
        self.assertEqual(len(result.get("WARNING", [])), 1)
        self.assertEqual(len(result.get("ERROR", [])), 1)

    def test_empty(self):
        self.assertEqual(parse_log_levels(""), {})

    def test_no_matches(self):
        self.assertEqual(parse_log_levels("just plain text"), {})


if __name__ == "__main__":
    unittest.main()
