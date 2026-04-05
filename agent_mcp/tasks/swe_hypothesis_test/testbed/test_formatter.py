import unittest
from formatter import format_report


class TestFormatReport(unittest.TestCase):
    def test_simple(self):
        result = format_report({"name": "Alice", "score": 95})
        self.assertIn("Name: Alice", result)
        self.assertIn("Score: 95", result)

    def test_list_values(self):
        result = format_report({"tags": ["python", "coding"]})
        self.assertIn("Tags: python, coding", result)

    def test_underscore_keys(self):
        result = format_report({"first_name": "Bob"})
        self.assertIn("First Name: Bob", result)


if __name__ == "__main__":
    unittest.main()
