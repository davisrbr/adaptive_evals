import unittest
from url_parser import extract_urls


class TestExtractUrls(unittest.TestCase):
    def test_simple_url(self):
        text = "Visit https://example.com/page for info"
        result = extract_urls(text)
        self.assertEqual(len(result), 1)
        self.assertIn("https://example.com/page", result[0])

    def test_query_params(self):
        text = "Click https://api.example.com/search?q=hello&lang=en here"
        result = extract_urls(text)
        self.assertEqual(len(result), 1)
        self.assertIn("?q=hello&lang=en", result[0])

    def test_fragment(self):
        text = "See https://docs.example.com/guide#installation"
        result = extract_urls(text)
        self.assertEqual(len(result), 1)
        self.assertIn("#installation", result[0])

    def test_no_urls(self):
        self.assertEqual(extract_urls("no urls here"), [])


if __name__ == "__main__":
    unittest.main()
