#!/usr/bin/env python3

import unittest
from pathlib import Path

import yaml
from numpy.testing import assert_allclose

import figure_comp.coordinate_tracking as ct
import figure_comp.structure_comp as sc
import figure_comp.structure_parse as sp
from figure_comp.tests.test_coordinate_tracking import get_coords


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

        # Any strings are presumed to be file paths
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

        pos_test = sp._parse_section(test_config).assemble_figure()
        pos_expected = [ct.Pos(path=p.resolve()) for p in [im_one, im_two]]

        # Test outer layer
        self.assertTrue(isinstance(pos_test, sc.Row))

        # Test that arrays are the same
        for im_test, im_expected in zip(pos_test.cont, pos_expected):
            self.assertEqual(im_test, im_expected)

        self.assertEqual(len(pos_test.cont), len(pos_expected))

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

        ims = [ct.Pos(path=p.resolve()) for p in im_paths]

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
        ims = [ct.Pos(path=p.resolve()) for p in im_paths]

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
        ims = [ct.Pos(path=p.resolve()) for p in im_paths]

        test_yaml = f"""
        - Row:
          - Col: [{im_paths[0]}, {im_paths[1]}]
          - {im_paths[4]}
          - Col: [{im_paths[2]}, {im_paths[3]}]
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        # test outer layer
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
            "tests/test_im/rect-im-3.png",
            "tests/test_im/rect-im-3.png",
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
          - {im_paths[4]}
          - {im_paths[4]}
          - Col:
            - {im_paths[2]}
            - {im_paths[3]}
          - Col:
            - {im_paths[2]}
            - {im_paths[3]}
            - {im_paths[6]}
          - options:
              y_size: 700
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        option_row_test = figure_test.y_size
        option_row_expected = 700

        option_col_test = figure_test[0].x_size
        option_col_expected = 750

        self.assertEqual(option_col_test, option_col_expected)
        self.assertEqual(option_row_test, option_row_expected)
        assembled_fig = figure_test.run()

        # figure_test.run().save("/tmp/rect-plot.png")
        #

    def test_nested_three_level(self):
        """ Read options in a nested element. """
        paths = [
            "tests/test_im/rect-im-1.png",
            "tests/test_im/rect-im-2.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
            "tests/test_im/square-im-5.png",
            "tests/test_im/rect-im-3.png",
            "tests/test_im/rect-im-3.png",
        ]
        im_paths = [Path(p) for p in paths]

        test_yaml = f"""
        - Row:
          - Col:
            - {im_paths[0]}
            - {im_paths[1]}
            - Row:
                - {im_paths[2]}
                - {im_paths[3]}
          - {im_paths[4]}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        assembled_fig = figure_test.run()
        # assembled_fig.sketch("/tmp/assembled-three.png", label="short")

    def test_nested_four_level(self):
        """ Read options in a nested element. """
        paths = [
            "tests/test_im/rect-im-1.png",
            "tests/test_im/rect-im-2.png",
            "tests/test_im/rect-im-3.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
            "tests/test_im/square-im-5.png",
            "tests/test_im/rect-im-3.png",
        ]
        im_paths = [Path(p) for p in paths]

        test_yaml = f"""
        - Row:
          - Col:
            - {im_paths[0]}
            - Row:
                - {im_paths[3]}
                - Col:
                    - {im_paths[0]}
                    - {im_paths[2]}
          - {im_paths[4]}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp._parse_section(test_config).assemble_figure()

        assembled_fig = figure_test.run()
        # assembled_fig.sketch("/tmp/assembled.png", label="short")
        # assembled_fig.populate("/tmp/nested-four-level.png")

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

    def test_tri_array_merge(self):
        """ Investigate rescaling of PosArray """
        test_yaml = """
        - Row:
          - 1.png
          - Col:
            - 2.png
            - Row:
              - 3.png
              - 4.png
        """

        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_structure = sp._parse_section(test_config).assemble_figure()
        pos_arr = figure_structure.run()

        # Widths of the squares in decreasing size
        full_width = 50
        med_width = 50 * (2 / 3)
        sma_width = med_width / 2

        x_test = get_coords(pos_arr, "x")
        x_expected = [0, full_width, full_width, full_width + sma_width]
        assert_allclose(x_test, x_expected)

        y_test = get_coords(pos_arr, "y")
        y_expected = [0, 0, med_width, med_width]
        assert_allclose(y_test, y_expected)

        dx_test = get_coords(pos_arr, "dx")
        dx_expected = [full_width, med_width, sma_width, sma_width]
        assert_allclose(dx_test, dx_expected)

        dy_test = get_coords(pos_arr, "dy")
        dy_expected = [full_width, med_width, sma_width, sma_width]
        assert_allclose(dy_test, dy_expected)


if __name__ == "__main__":
    unittest.main()
