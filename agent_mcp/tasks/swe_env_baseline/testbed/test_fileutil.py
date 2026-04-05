import unittest
from fileutil import find_configs


class TestFindConfigs(unittest.TestCase):
    def test_finds_nested(self):
        configs = find_configs("/testbed/configs")
        self.assertEqual(len(configs), 2)  # main.yaml + sub/extra.yaml
        self.assertTrue(any("main.yaml" in c for c in configs))
        self.assertTrue(any("extra.yaml" in c for c in configs))


if __name__ == "__main__":
    unittest.main()
