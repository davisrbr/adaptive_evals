import unittest
from phone_parser import find_phone_numbers


class TestFindPhoneNumbers(unittest.TestCase):
    def test_parens_format(self):
        result = find_phone_numbers("Call (123) 456-7890 now")
        self.assertEqual(len(result), 1)

    def test_dash_format(self):
        result = find_phone_numbers("Call 123-456-7890 now")
        self.assertEqual(len(result), 1)

    def test_both_formats(self):
        result = find_phone_numbers("(111) 222-3333 and 444-555-6666")
        self.assertEqual(len(result), 2)

    def test_no_phones(self):
        self.assertEqual(find_phone_numbers("no phone here"), [])


if __name__ == "__main__":
    unittest.main()
