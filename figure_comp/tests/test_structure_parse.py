#!/usr/bin/env python3

import tempfile
import unittest

import yaml
from icecream import ic

import figure_comp.structure_comp as sc
import figure_comp.structure_parse as sp


class TestYamlParsing(unittest.TestCase):
    def test_manual_simple_parse(self):
        """ Experiment with a simple yaml structure. """

        test_yaml = """
        - Row:
          - /path/one
          - /path/two
          - options:
             max_size: 20
             new_size: 45
        """

        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)[0]

        self.assertTrue("Row" in test_config)

        # Any strings are persumed to be file paths
        row_info = test_config["Row"]
        paths_test = [r for r in row_info if isinstance(r, str)]
        paths_expected = ["/path/one", "/path/two"]
        self.assertEqual(paths_test, paths_expected)

        # Now read the options
        for row in row_info:
            if not isinstance(row, str) and "options" in row:
                options_test = row["options"]
                break
        else:
            raise AssertionError("Unable to read options from test yaml")

        options_expected = dict(max_size=20, new_size=45)
        self.assertEqual(options_test, options_expected)

    def test_simple_parse(self):
        """ Experiment with a simple yaml structure. """
        test_yaml = """
        - Row:
          - /path/one
          - /path/two
          - options:
             max_size: 20
             new_size: 45
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        leaf_test = sp._parse_section(test_config)

        leaf_expected = (
            sc.Row,
            ["/path/one", "/path/two"],
            dict(max_size=20, new_size=45),
        )

        self.assertEqual(leaf_test, leaf_expected)

    def test_nested_parse(self):
        """ A structure with a column within a row. """
        test_yaml = """
        - Row:
          - /path/one
          - /path/two
          - Col:
            - /path/three
            - /path/four
          - options:
             max_size: 20
             new_size: 45
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        leaf_test = sp._parse_section(test_config)

        nested_leaf = (sc.Col, ["/path/three", "/path/four"], {})
        leaf_expected = (
            sc.Row,
            ["/path/one", "/path/two", nested_leaf],
            dict(max_size=20, new_size=45),
        )

        self.assertEqual(leaf_test, leaf_expected)

    def test_nested_double_parse(self):
        """ A structure with two columns within a row. """
        test_yaml = """
        - Row:
          - Col:
            - /path/one
            - /path/two
          - /path/five
          - Col:
            - /path/three
            - /path/four
            - options:
                width: 15
          - options:
             max_size: 20
             new_size: 45
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        leaf_test = sp._parse_section(test_config)

        first_col = (sc.Col, ["/path/one", "/path/two"], {})
        second_col = (sc.Col, ["/path/three", "/path/four"], {"width": 15})
        leaf_expected = (
            sc.Row,
            [first_col, "/path/five", second_col],
            {"max_size": 20, "new_size": 45},
        )

        self.assertEqual(leaf_test, leaf_expected)


if __name__ == "__main__":
    unittest.main()
