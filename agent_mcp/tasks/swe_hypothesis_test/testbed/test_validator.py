import unittest
from validator import validate_email


class TestValidateEmail(unittest.TestCase):
    def test_simple_valid(self):
        self.assertTrue(validate_email("user@example.com"))

    def test_plus_addressing(self):
        self.assertTrue(validate_email("user+tag@example.com"))

    def test_subdomain(self):
        self.assertTrue(validate_email("user@mail.example.com"))

    def test_invalid_no_at(self):
        self.assertFalse(validate_email("userexample.com"))

    def test_invalid_no_domain(self):
        self.assertFalse(validate_email("user@"))

    def test_dots_and_hyphens(self):
        self.assertTrue(validate_email("first.last@my-company.org"))


if __name__ == "__main__":
    unittest.main()
