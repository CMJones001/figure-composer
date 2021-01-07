#!/usr/bin/env python3

import tempfile
import unittest

import yaml
from icecream import ic
from pathlib import Path
from skimage.io import imread
from numpy.testing import assert_allclose

import figure_comp.structure_comp as sc
import figure_comp.structure_parse as sp
import figure_comp.figure_rescale as fr


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
        figure_test = sp._parse_section(test_config).as_tuple()

        leaf_expected = (
            sc.Row,
            ["/path/one", "/path/two"],
            dict(max_size=20, new_size=45),
        )

        self.assertEqual(figure_test, leaf_expected)

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
        figure_test = sp._parse_section(test_config).as_tuple()

        nested_leaf = (sc.Col, ["/path/three", "/path/four"], {})
        leaf_expected = (
            sc.Row,
            ["/path/one", "/path/two", nested_leaf],
            dict(max_size=20, new_size=45),
        )

        self.assertEqual(figure_test, leaf_expected)

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
        figure_test = sp._parse_section(test_config).as_tuple()

        first_col = (sc.Col, ["/path/one", "/path/two"], {})
        second_col = (sc.Col, ["/path/three", "/path/four"], {"width": 15})
        leaf_expected = (
            sc.Row,
            [first_col, "/path/five", second_col],
            {"max_size": 20, "new_size": 45},
        )

        self.assertEqual(figure_test, leaf_expected)


class TestAssembleStruct(unittest.TestCase):
    def test_simple_parse(self):
        """ Assemble an image from a simple structure. """
        im_one = Path("tests/test_im/square-im-1.png")
        im_two = Path("tests/test_im/square-im-2.png")
        test_yaml = f"""
        - Row:
          - {im_one}
          - {im_two}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        ims = [imread(p) for p in [im_one, im_two]]

        # Test outer layer
        self.assertTrue(isinstance(figure_test, sc.Row))

        # Test that arrays are the same
        for im_test, im_expected in zip(figure_test.cont, ims):
            assert_allclose(im_test.data, im_expected)

        self.assertEqual(len(figure_test.cont), len(ims))

    def test_nested_parse(self):
        """ Assemble an image from a nested structure. """
        paths = [
            "tests/test_im/square-im-1.png",
            "tests/test_im/square-im-2.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
        ]
        im_paths = [Path(p) for p in paths]

        test_yaml = f"""
        - Row:
          - {im_paths[0]}
          - {im_paths[1]}
          - Col:
            - {im_paths[2]}
            - {im_paths[3]}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        ims = [fr.Image(imread(p), p) for p in im_paths]

        # Test outer layer
        self.assertTrue(isinstance(figure_test, sc.Row))

        self.assertEqual(figure_test.cont[0], ims[0])
        self.assertEqual(figure_test.cont[1], ims[1])

        inner_col = figure_test.cont[2]
        self.assertEqual(inner_col.cont[0], ims[2])
        self.assertEqual(inner_col.cont[1], ims[3])

    def test_nested_double_parse(self):
        """ Assemble a structure with two columns within a row. """
        paths = [
            "tests/test_im/square-im-1.png",
            "tests/test_im/square-im-2.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
            "tests/test_im/square-im-5.png",
        ]
        im_paths = [Path(p) for p in paths]
        ims = [fr.Image(imread(p), p) for p in im_paths]

        test_yaml = f"""
        - Row:
          - Col:
            - {im_paths[0]}
            - {im_paths[1]}
          - {im_paths[4]}
          - Col:
            - {im_paths[2]}
            - {im_paths[3]}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        # Test outer layer
        self.assertTrue(isinstance(figure_test, sc.Row))

        # Middle column with single image
        self.assertEqual(figure_test[1], ims[4])

        # First column
        self.assertEqual(figure_test[0][0], ims[0])
        self.assertEqual(figure_test[0][1], ims[1])

        # Final column
        self.assertEqual(figure_test[2][0], ims[2])
        self.assertEqual(figure_test[2][1], ims[3])

    def test_nested_double_short(self):
        """ Assemble a structure with two columns within a row using shortened notation. """
        paths = [
            "tests/test_im/square-im-1.png",
            "tests/test_im/square-im-2.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
            "tests/test_im/square-im-5.png",
        ]
        im_paths = [Path(p) for p in paths]
        ims = [fr.Image(imread(p), p) for p in im_paths]

        test_yaml = f"""
        - Row:
          - Col: [{im_paths[0]}, {im_paths[1]}]
          - {im_paths[4]}
          - Col: [{im_paths[2]}, {im_paths[3]}]
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        # Test outer layer
        self.assertTrue(isinstance(figure_test, sc.Row))

        # Middle column with single image
        self.assertEqual(figure_test[1], ims[4])

        # First column
        self.assertEqual(figure_test[0][0], ims[0])
        self.assertEqual(figure_test[0][1], ims[1])

        # Final column
        self.assertEqual(figure_test[2][0], ims[2])
        self.assertEqual(figure_test[2][1], ims[3])

    def test_simple_parse_options(self):
        """ Read options in the simplest case. """
        im_one = Path("tests/test_im/square-im-1.png")
        im_two = Path("tests/test_im/square-im-2.png")
        test_yaml = f"""
        - Row:
          - {im_one}
          - {im_two}
          - options:
              y_size: 600
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        self.assertEqual(figure_test.y_size, 600)

    def test_nested_rect_parse_options(self):
        """ Read options in a nested element. """
        paths = [
            "tests/test_im/rect-im-1.png",
            "tests/test_im/rect-im-2.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
            "tests/test_im/square-im-5.png",
        ]
        im_paths = [Path(p) for p in paths]

        test_yaml = f"""
        - Row:
          - Col:
            - {im_paths[0]}
            - {im_paths[1]}
            - options:
                x_size: 750
          - {im_paths[4]}
          - Col:
            - {im_paths[2]}
            - {im_paths[3]}
          - options:
              y_size: 700
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        option_row_test = figure_test.y_size
        option_row_expected = 700

        option_col_test = figure_test[0].x_size
        option_col_expected = 750

        self.assertEqual(option_col_test, option_col_expected)
        self.assertEqual(option_row_test, option_row_expected)

        # figure_test.run().save("/tmp/rect-plot.png")

    def test_simple_parse_flow(self):
        """ Read options in the simplest case with the space saving format. """
        im_one = Path("tests/test_im/square-im-1.png")
        im_two = Path("tests/test_im/square-im-2.png")
        test_yaml = f"""
        - Row:
          - {im_one}
          - {im_two}
          - options: {{y_size: 600}}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        self.assertEqual(figure_test.y_size, 600)


if __name__ == "__main__":
    unittest.main()
