#!/usr/bin/env python3

import unittest
from pathlib import Path

import numpy as np
import yaml
from icecream import ic
from numpy.testing import assert_allclose

import figure_comp.coordinate_tracking as ct
import figure_comp.structure_comp as sc
import figure_comp.structure_parse as sp
from figure_comp.load_image import Label
from figure_comp.tests.test_coordinate_tracking import get_coords

project_dir = Path(__file__).resolve().parent
test_im_dir = project_dir / "test_im/"

paths = [
    test_im_dir / "square-im-1.png",
    test_im_dir / "square-im-2.png",
    test_im_dir / "square-im-3.png",
    test_im_dir / "square-im-4.png",
    test_im_dir / "square-im-5.png",
    test_im_dir / "rect-im-1.png",
    test_im_dir / "rect-im-2.png",
    test_im_dir / "rect-im-3.png",
]
im_paths = [Path(p) for p in paths]


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
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp.parse_yaml(test_config, dry=True)

        leaf_expected = (
            "Row",
            [ct.Pos("/path/one"), ct.Pos("/path/two")],
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
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp.parse_yaml(test_config, dry=True)

        nested_leaf = ("Col", [ct.Pos("/path/three"), ct.Pos("/path/four")])
        leaf_expected = (
            "Row",
            [ct.Pos("/path/one"), ct.Pos("/path/two"), nested_leaf],
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
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp.parse_yaml(test_config, dry=True)

        first_col = ("Col", [ct.Pos("/path/one"), ct.Pos("/path/two")])
        second_col = ("Col", [ct.Pos("/path/three"), ct.Pos("/path/four")])
        leaf_expected = (
            "Row",
            [first_col, ct.Pos("/path/five"), second_col],
        )

        self.assertEqual(figure_test, leaf_expected)


class TestAssembleStruct(unittest.TestCase):
    def test_simple_parse(self):
        """ Assemble an image from a simple structure. """
        im_one = Path("tests/test_im/square-im-1.png").resolve()
        im_two = Path("tests/test_im/square-im-2.png").resolve()
        test_yaml = f"""
        - Row:
          - {im_one}
          - {im_two}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)

        pos_test = sp.parse_yaml(test_config)
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
        im_paths = [Path(p).resolve() for p in paths]

        test_yaml = f"""
        - Row:
          - {im_paths[0]}
          - {im_paths[1]}
          - Col:
            - {im_paths[2]}
            - {im_paths[3]}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp.parse_yaml(test_config, dry=True)

        ims = [ct.Pos(path=p.resolve()) for p in im_paths]

        # Test outer layer
        header_test = figure_test[0]
        header_expected = "Row"
        self.assertEqual(header_test, header_expected)
        body = figure_test[1]

        self.assertEqual(body[0], ims[0])
        self.assertEqual(body[1], ims[1])

        inner_col = body[2][1]
        self.assertEqual(inner_col[0], ims[2])
        self.assertEqual(inner_col[1], ims[3])

    def test_nested_double_parse(self):
        """ Assemble a structure with two columns within a row. """
        paths = [
            "tests/test_im/square-im-1.png",
            "tests/test_im/square-im-2.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
            "tests/test_im/square-im-5.png",
        ]
        im_paths = [Path(p).resolve() for p in paths]
        ims = [ct.Pos(path=p) for p in im_paths]

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
        figure_test = sp.parse_yaml(test_config, dry=True)

        # Middle column with single image
        main_body = figure_test[1]
        self.assertEqual(main_body[1], ims[4])

        # First column
        col_one = main_body[0][1]
        self.assertEqual(col_one[0], ims[0])
        self.assertEqual(col_one[1], ims[1])

        # Final column
        col_two = main_body[2][1]
        self.assertEqual(col_two[0], ims[2])
        self.assertEqual(col_two[1], ims[3])

    def test_nested_double_short(self):
        """ Assemble a structure with two columns within a row using shortened notation. """
        paths = [
            "tests/test_im/square-im-1.png",
            "tests/test_im/square-im-2.png",
            "tests/test_im/square-im-3.png",
            "tests/test_im/square-im-4.png",
            "tests/test_im/square-im-5.png",
        ]
        im_paths = [Path(p).resolve() for p in paths]
        ims = [ct.Pos(path=p) for p in im_paths]

        test_yaml = f"""
        - Row:
          - Col: [{im_paths[0]}, {im_paths[1]}]
          - {im_paths[4]}
          - Col: [{im_paths[2]}, {im_paths[3]}]
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp.parse_yaml(test_config, dry=True)

        # Middle column with single image
        main_body = figure_test[1]
        self.assertEqual(main_body[1], ims[4])

        # First column
        col_one = main_body[0][1]
        self.assertEqual(col_one[0], ims[0])
        self.assertEqual(col_one[1], ims[1])

        # Final column
        col_two = main_body[2][1]
        self.assertEqual(col_two[0], ims[2])
        self.assertEqual(col_two[1], ims[3])

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
        figure_test = sp.parse_yaml(test_config)

    def test_simple_parse_flow(self):
        """ Read options in the simplest case with the space saving format. """
        im_one = Path("tests/test_im/square-im-1.png")
        im_two = Path("tests/test_im/square-im-2.png")
        test_yaml = f"""
        - Row:
          - {im_one}: {{text: "test-label"}}
          - {im_two}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        figure_test = sp.parse_yaml(test_config)

        label_expected = "test-label"
        label_test = figure_test[0].label.text
        self.assertEqual(label_test, label_expected)

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
        pos_arr = sp.parse_yaml(test_config).run()

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


class TestAssembleOptions(unittest.TestCase):
    """ Test parsing of options and figure labels. """

    def test_simple_parse_options(self):
        """ Read options in the simplest case. """
        test_yaml = f"""
        - Row:
          - {paths[0]}:
              text: "a."
              pos: (0.1, 0.1)
          - Col:
            - {paths[1]}
            - {paths[2]}
            - Row:
              - {paths[3]}
              - {paths[4]}
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)[0]

        def _parse_complex_path(leaf):
            """ Parse a path with labels: """
            for path, labels in leaf.items():
                if "pos" in labels:
                    pos_str = labels["pos"].strip("()")
                    pos = np.fromstring(pos_str, sep=", ")
                    labels.pop("pos")
                else:
                    pos = None
                label = Label(**labels, pos=pos)
                return ct.Pos(path, label)

        def _parse_path(leaf):
            """ Parse a simple path into a path. """
            return ct.Pos(leaf)

        def _is_subbranch(leaf):
            return "Col" in leaf or "Row" in leaf

        def read_branch(branch, dry=False):
            struct = []
            header = "Row" if "Row" in branch else "Col"
            header_struct = sc.Row if header == "Row" else sc.Col

            for entry in branch[header]:
                if isinstance(entry, str):
                    struct.append(_parse_path(entry))
                elif _is_subbranch(entry):
                    struct.append(read_branch(entry, dry=dry))
                elif isinstance(entry, dict):
                    struct.append(_parse_complex_path(entry))
                else:
                    raise ValueError("Unable to parse branch")

            if dry:
                return (header, struct)
            return header_struct(struct)

        struct = read_branch(test_config).run()
        struct.populate("/tmp/new_parse.png")

    def test_parse_global_opts(self):
        """ Test that we can read global options. """
        test_yaml = f"""
        - Row:
          - {paths[0]}
          - Col:
            - {paths[1]}
            - {paths[2]}
            - Row:
              - {paths[3]}
              - {paths[4]}
        - Options:
          - size: 45
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        pos_arr = sp.parse_yaml(test_config).run()

        labels = get_coords(pos_arr, "label")
        sizes_test = [l.size for l in labels]
        sizes_expected = np.ones(5) * 45

        assert_allclose(sizes_test, sizes_expected)

    def test_parse_global_override(self):
        """ Test that we can override a global options. """
        test_yaml = f"""
        - Row:
          - {paths[0]}
          - Col:
            - {paths[1]}: {{size: 10}}
            - {paths[2]}
            - Row:
              - {paths[3]}
              - {paths[4]}
        - Options:
          - size: 45
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        pos_arr = sp.parse_yaml(test_config).run()

        labels = get_coords(pos_arr, "label")
        sizes_test = [l.size for l in labels]
        sizes_expected = np.ones(5) * 45
        sizes_expected[1] = 10

        assert_allclose(sizes_test, sizes_expected)

    def test_parse_global_labels(self):
        """ Test that we parse the default label gen. """
        test_yaml = f"""
        - Row:
          - {paths[0]}
          - Col:
            - {paths[1]}
            - {paths[2]}
            - Row:
              - {paths[3]}
              - {paths[4]}
        - Options:
          - default_label: "{{index+1}}."
          - size: 18
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        pos_arr = sp.parse_yaml(test_config).run()

        labels = get_coords(pos_arr, "label")
        text_test = [l.text for l in labels]
        text_expected = [f"{i+1}." for i in range(5)]

        self.assertEqual(text_test, text_expected)

        sizes_test = [l.size for l in labels]
        sizes_expected = np.ones(5) * 18
        assert_allclose(sizes_test, sizes_expected)

    def test_parse_global_label_override(self):
        """ Test that we parse the default label gen with an override. """
        test_yaml = f"""
        - Row:
          - {paths[0]}
          - Col:
            - {paths[1]}
            - {paths[2]}:
                text: "A"
            - Row:
              - {paths[3]}
              - {paths[4]}
        - Options:
          - default_label: "{{index+1}}."
        """
        test_config = yaml.load(test_yaml, Loader=yaml.FullLoader)
        pos_arr = sp.parse_yaml(test_config).run()

        labels = get_coords(pos_arr, "label")
        text_test = [l.text for l in labels]
        text_expected = [f"{i+1}." for i in range(5)]
        text_expected[2] = "A"

        self.assertEqual(text_test, text_expected)


if __name__ == "__main__":
    unittest.main()
