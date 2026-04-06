import unittest
from csvparser import parse_csv


class TestParseCsv(unittest.TestCase):
    def test_simple(self):
        text = "name,age\nAlice,30\nBob,25"
        result = parse_csv(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Alice")
        self.assertEqual(result[0]["age"], "30")

    def test_quoted_fields(self):
        text = 'name,city\nAlice,"New York, NY"\nBob,"San Francisco, CA"'
        result = parse_csv(text)
        self.assertEqual(result[0]["city"], "New York, NY")
        self.assertEqual(result[1]["city"], "San Francisco, CA")

    def test_empty(self):
        self.assertEqual(parse_csv(""), [])
        self.assertEqual(parse_csv("headers_only"), [])


if __name__ == "__main__":
    unittest.main()
