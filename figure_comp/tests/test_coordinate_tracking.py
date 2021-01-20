#!/usr/bin/env python3

import itertools
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from figure_comp.coordinate_tracking import Pos, PosArray
import figure_comp.coordinate_tracking as ct


def create_row_array(x_size=50, num=5, x_offset=0, y_offset=0) -> PosArray:
    """ Create a simple row of images that are ``x_size`` wide. """
    x_pos = np.arange(0, num) * x_size + x_offset
    pos_arr = PosArray([Pos(x_size, x_size, x, y_offset, path=None) for x in x_pos])
    return pos_arr


def create_pos_array(
    x_size=50, y_size=50, x_num=1, y_num=1, x_offset=0, y_offset=0
) -> PosArray:
    """ General creation of PosArray. """
    x_pos = np.arange(0, x_num) * x_size + x_offset
    y_pos = np.arange(0, y_num) * y_size + y_offset
    pos_arr = PosArray(
        [
            Pos(x_size, y_size, x, y, path=None)
            for x, y in itertools.product(x_pos, y_pos)
        ]
    )
    return pos_arr


def create_pos_array_opts(
    x_size=50, y_size=50, x_num=1, y_num=1, paths=None, opts=None
) -> PosArray:
    """ General creation of PosArray. """
    x_pos = np.arange(0, x_num) * x_size
    y_pos = np.arange(0, y_num) * y_size

    paths = [] if paths is None else paths
    opts = {} if opts is None else opts
    x_y_prod = itertools.product(x_pos, y_pos)
    full_iter = itertools.zip_longest(x_y_prod, paths, opts)

    pos_arr = PosArray(
        [Pos(x_size, y_size, x, y, path, opts) for (x, y), path, opts in full_iter]
    )
    return pos_arr


def get_coords(arr: PosArray, attr: str):
    """ Get the given coordinates/attribute list from a nested set of ``PosArray``s. """
    coord_list = []
    for p in arr:
        if p.is_array:
            coord_list += get_coords(p, attr)
        else:
            coord_list.append(getattr(p, attr))
    return coord_list


class TestCreatePos(unittest.TestCase):
    """ Metatest for the create_pos_array helper function. """

    def validate_pos_array(self, func_kwargs, len_expected, shape_expected):
        """ Test the properties of the generated PosArray. """
        pos_arr = create_pos_array(**func_kwargs)

        x_min_test = pos_arr.x_min
        y_min_test = pos_arr.y_min
        x_max_test = pos_arr.x_max
        y_max_test = pos_arr.y_max

        x_min_expected, y_min_expected, x_max_expected, y_max_expected = shape_expected

        self.assertEqual(x_min_test, x_min_expected)
        self.assertEqual(y_min_test, y_min_expected)
        self.assertEqual(x_max_test, x_max_expected)
        self.assertEqual(y_max_test, y_max_expected)

        len_test = len(pos_arr)
        self.assertEqual(len_test, len_expected)

    def test_single_shape(self):
        """ Create a single square box. """
        self.validate_pos_array({}, 1, (0, 0, 50, 50))

    def test_row_shape(self):
        """ Create a single row. """
        self.validate_pos_array({"x_num": 3}, 3, (0, 0, 150, 50))

    def test_col_shape(self):
        """ Create a single col. """
        self.validate_pos_array({"y_num": 4}, 4, (0, 0, 50, 200))

    def test_col_shape_offset(self):
        """ Create a single col with offset. """
        self.validate_pos_array({"y_num": 4, "y_offset": 50}, 4, (0, 50, 50, 250))

    def test_grid_shape(self):
        """ Create a single col with offset. """
        self.validate_pos_array(
            {"y_num": 4, "y_offset": 50, "x_num": 3}, 12, (0, 50, 150, 250)
        )


class TestScaling(unittest.TestCase):
    """ Test the simple pos manipulations. """

    def test_shift_single_x(self):
        """ Move multiple images. """
        count = 5
        x_size = 50
        pos_arr = create_row_array(x_size=x_size, num=count)
        x_move = 25
        pos_arr.shift_x(x_move)

        x_expected = np.arange(0, count) * x_size + x_move
        x_test = [p.x for p in pos_arr]

        assert_allclose(x_test, x_expected)

    def test_shift_again_x(self):
        """ Move multiple images, twice. """
        count = 5
        x_size = 50
        pos_arr = create_row_array(x_size=x_size, num=count)
        x_move = 25
        pos_arr.shift_x(x_move)
        pos_arr.shift_x(x_move)

        x_expected = np.arange(0, count) * x_size + x_move * 2
        x_test = [p.x for p in pos_arr]

        assert_allclose(x_test, x_expected)

    def test_scaling_simple(self):
        """ Halve the size of three images. """
        scale_factor = 0.5
        count = 5
        x_size = 50
        pos_arr = create_row_array(x_size=x_size, num=count)
        pos_arr.rescale(scale_factor)

        # Start pos in the x direction should also be affected
        x_expected = np.arange(0, count) * x_size * scale_factor
        x_test = [p.x for p in pos_arr]
        assert_allclose(x_test, x_expected)

        # Y starting pos should still be zero
        y_expected = x_expected * 0
        y_test = [p.y for p in pos_arr]
        assert_allclose(y_test, y_expected)

        # Ensure the others are as we expect
        dx_expected = x_size * scale_factor
        dy_expected = x_size * scale_factor
        dx_test = [p.dx for p in pos_arr]
        dy_test = [p.dy for p in pos_arr]

        assert_allclose(dx_test, dx_expected)
        assert_allclose(dy_test, dy_expected)

    def test_scaling_offset(self):
        """ Halve the size of three images. """
        scale_factor = 0.5
        count = 5
        x_size = 50
        x_offset = 30
        pos_arr = create_row_array(x_size=x_size, num=count, x_offset=x_offset)
        pos_arr.rescale(scale_factor)

        # Start pos in the x direction should also be affected
        x_expected = np.arange(0, count) * x_size * scale_factor + x_offset
        x_test = [p.x for p in pos_arr]
        assert_allclose(x_test, x_expected)

        # Y starting pos should still be zero
        y_expected = x_expected * 0
        y_test = [p.y for p in pos_arr]
        assert_allclose(y_test, y_expected)

        # Ensure the others are as we expect
        dx_expected = x_size * scale_factor
        dy_expected = x_size * scale_factor
        dx_test = [p.dx for p in pos_arr]
        dy_test = [p.dy for p in pos_arr]

        assert_allclose(dx_test, dx_expected)
        assert_allclose(dy_test, dy_expected)


class TestStacking(unittest.TestCase):
    """ Test the top level merging functions. """

    def test_col_two_add(self):
        """ Check adding of two column shapes. """
        count_right = 4
        col_right = create_pos_array(y_num=count_right)

        count_left = 3
        col_left = create_pos_array(y_num=count_left)

        pos_arr = col_left.stack_right(col_right)

        x_max_expected = 50 * (1.0 + count_left / count_right)
        x_max_test = pos_arr.x_max
        self.assertEqual(x_max_test, x_max_expected)

        y_max_expected = 50 * (count_left)
        y_max_test = pos_arr.y_max
        self.assertEqual(y_max_test, y_max_expected)

    def test_col_three_add(self):
        """ Check adding of two column shapes and a row. """
        count_right = 4
        col_right = create_pos_array(y_num=count_right)

        count_left = 3
        col_left = create_pos_array(y_num=count_left)

        count_row = 2
        row = create_pos_array(x_num=count_row)

        pos_arr = col_left.stack_right(col_right).stack_right(row)

        x_max_expected = 50 * (1.0 + count_left / count_right + count_left * count_row)
        x_max_test = pos_arr.x_max
        self.assertEqual(x_max_test, x_max_expected)

        y_max_expected = 50 * (count_left)
        y_max_test = pos_arr.y_max
        self.assertEqual(y_max_test, y_max_expected)

    def test_col_three_add_alt(self):
        """ Check adding of two coulmn shapes and a row using operator overloading. """
        count_right = 4
        col_right = create_pos_array(y_num=count_right)

        count_left = 3
        col_left = create_pos_array(y_num=count_left)

        count_row = 2
        row = create_pos_array(x_num=count_row)

        pos_arr = col_left + col_right + row

        x_max_expected = 50 * (1.0 + count_left / count_right + count_left * count_row)
        x_max_test = pos_arr.x_max
        assert_allclose(x_max_test, x_max_expected)

        y_max_expected = 50 * (count_left)
        y_max_test = pos_arr.y_max
        assert_allclose(y_max_test, y_max_expected)

    def test_col_three_add_reduce(self):
        """ Check adding of two coulmn shapes and a row using operator overloading. """
        count_right = 4
        col_right = create_pos_array(y_num=count_right)

        count_left = 3
        col_left = create_pos_array(y_num=count_left)

        count_row = 2
        row = create_pos_array(x_num=count_row)

        structs = [col_left, col_right, row]
        pos_arr = ct.merge_row(structs)

        x_max_expected = 50 * (1.0 + count_left / count_right + count_left * count_row)
        x_max_test = pos_arr.x_max
        assert_allclose(x_max_test, x_max_expected)

        y_max_expected = 50 * (count_left)
        y_max_test = pos_arr.y_max
        assert_allclose(y_max_test, y_max_expected)

    def test_row_two_add(self):
        """ Check adding of two row shapes. """
        count_top = 2
        row_top = create_pos_array(x_num=count_top)

        count_bottom = 3
        row_bottom = create_pos_array(x_num=count_bottom)

        pos_arr = row_top.stack_below(row_bottom)

        x_max_expected = 50 * (count_top)
        x_max_test = pos_arr.x_max
        assert_allclose(x_max_test, x_max_expected)

        y_max_expected = 50 * (1 + count_top / count_bottom)
        y_max_test = pos_arr.y_max
        assert_allclose(y_max_test, y_max_expected)

    def test_row_two_add_alt(self):
        """ Check adding of two row shapes. """
        count_top = 5
        row_top = create_pos_array(x_num=count_top)

        count_bottom = 3
        row_bottom = create_pos_array(x_num=count_bottom)

        pos_arr = row_top / row_bottom

        x_max_expected = 50 * (count_top)
        x_max_test = pos_arr.x_max
        assert_allclose(x_max_test, x_max_expected)

        y_max_expected = 50 * (1 + count_top / count_bottom)
        y_max_test = pos_arr.y_max
        assert_allclose(y_max_test, y_max_expected)


class TestStackingIrregular(unittest.TestCase):
    """ Test the top level merging functions on rectangular shapes. """

    def test_single_col_merge(self):
        """ Stack the shapes into a simple column. """
        x_size = 80
        y_size = 40
        count = 4

        pos_arr = create_pos_array(x_size, y_size, y_num=count)

        x_test = get_coords(pos_arr, "x")
        x_expected = np.arange(count) * 0
        assert_allclose(x_test, x_expected)

        y_test = get_coords(pos_arr, "y")
        y_expected = np.arange(count) * y_size
        assert_allclose(y_test, y_expected)

        dx_test = get_coords(pos_arr, "dx")
        dx_expected = np.ones(count) * x_size
        assert_allclose(dx_test, dx_expected)

        dy_test = get_coords(pos_arr, "dy")
        dy_expected = np.ones(count) * y_size
        assert_allclose(dy_test, dy_expected)

    def test_col_two_add(self):
        """ Check adding of two column shapes. """
        x_one_size = 60
        y_one_size = 30
        count_one = 3
        pos_one = create_pos_array(x_one_size, y_one_size, y_num=count_one)
        aspect_one = x_one_size / y_one_size

        x_two_size = 30
        y_two_size = 30
        count_two = 2
        pos_two = create_pos_array(x_two_size, y_two_size, y_num=count_two)
        aspect_two = x_two_size / y_two_size

        pos_arr = pos_one + pos_two

        total_height_test = pos_arr.y_range
        total_height_expected = y_one_size * count_one
        self.assertEqual(total_height_test, total_height_expected)

        aspect_test = get_coords(pos_arr, "aspect")
        aspect_expected = [aspect_one] * count_one + [aspect_two] * count_two
        assert_allclose(aspect_test, aspect_expected)

    def test_col_three_add(self):
        """ Check adding of two column shapes and a row. """
        x_one_size = 60
        y_one_size = 30
        count_one = 3
        pos_one = create_pos_array(x_one_size, y_one_size, y_num=count_one)
        aspect_one = x_one_size / y_one_size
        total_height = count_one * y_one_size

        x_two_size = 30
        y_two_size = 30
        count_two = 2
        pos_two = create_pos_array(x_two_size, y_two_size, y_num=count_two)
        aspect_two = x_two_size / y_two_size

        pos_arr = Pos(20, count_one * y_one_size) + (pos_one + pos_two)
        pos_arr.sketch("/tmp/test-figure.png", label="short")

        x_test = get_coords(pos_arr, "x")
        x_expected = [0, 20, 20, 20, 80, 80]
        assert_allclose(x_test, x_expected)

        y_test = get_coords(pos_arr, "y")
        y_expected = [0, 0, 30, 60, 0, 45]
        assert_allclose(y_test, y_expected)


class TestPosCombine(unittest.TestCase):
    """ Test the merging of the Pos and PosArray"""

    def test_pos_comp_row(self):
        """ Merging of multiple pos into a row. """
        count_row = 3
        positions = [Pos(50, 50) for i in range(count_row)]
        pos_arr = ct.merge_row(positions)

        len_test = len(pos_arr)
        len_expected = count_row
        self.assertEqual(len_test, len_expected)

        pos_expected = np.arange(count_row) * 50
        pos_test = [p.x for p in pos_arr]
        assert_allclose(pos_test, pos_expected)

    def test_pos_comp_col(self):
        """ Merging of multiple pos into a col. """
        count_row = 3
        positions = [Pos(50, 50) for i in range(count_row)]
        pos_arr = ct.merge_col(positions)

        len_test = len(pos_arr)
        len_expected = count_row
        self.assertEqual(len_test, len_expected)

        pos_expected = np.arange(count_row) * 50
        pos_test = [p.y for p in pos_arr]
        assert_allclose(pos_test, pos_expected)

        pos_expected = np.arange(count_row) * 0
        pos_test = [p.x for p in pos_arr]
        assert_allclose(pos_test, pos_expected)

    def test_pos_comp_merged(self):
        """ Merging of nested positions. """
        count_row = 3
        positions = [Pos(50, 50) for i in range(count_row)]
        pos_arr = ct.merge_row(
            [ct.merge_col([positions[0], positions[1]]), positions[2]]
        )

        len_test = len(pos_arr)
        len_expected = count_row
        self.assertEqual(len_test, len_expected)

        pos_expected = [50, 50, 100]
        pos_test = [p.dy for p in pos_arr]
        assert_allclose(pos_test, pos_expected)

        pos_expected = [50, 50, 100]
        pos_test = [p.dx for p in pos_arr]
        assert_allclose(pos_test, pos_expected)

    def test_tri_array_merge(self):
        """ Investigate rescaling of nested PosArray """
        pos_arr = Pos(50, 50) / (Pos(50, 50) + Pos(50, 50))
        pos_arr = Pos(75, 75) + pos_arr

        x_test = get_coords(pos_arr, "x")
        x_expected = [0, 75, 75, 100]
        assert_allclose(x_test, x_expected)

        y_test = get_coords(pos_arr, "y")
        y_expected = [0, 0, 50, 50]
        assert_allclose(y_test, y_expected)

        dx_test = get_coords(pos_arr, "dx")
        dx_expected = [75, 50, 25, 25]
        assert_allclose(dx_test, dx_expected)

        dy_test = get_coords(pos_arr, "dy")
        dy_expected = [75, 50, 25, 25]
        assert_allclose(dy_test, dy_expected)

    def test_quad_array_merge(self):
        """ Investigate rescaling of nested PosArray """
        pos_arr = Pos(100, 100) / (Pos(50, 50) + Pos(50, 50))
        pos_arr = (Pos(75, 75) / Pos(75, 75)) + pos_arr

        x_test = get_coords(pos_arr, "x")
        x_expected = [0, 0, 75, 75, 125]
        assert_allclose(x_test, x_expected)

        y_test = get_coords(pos_arr, "y")
        y_expected = [0, 75, 0, 100, 100]
        assert_allclose(y_test, y_expected)

        dx_test = get_coords(pos_arr, "dx")
        dx_expected = [75, 75, 100, 50, 50]
        assert_allclose(dx_test, dx_expected)

        dy_test = get_coords(pos_arr, "dy")
        dy_expected = [75, 75, 100, 50, 50]
        assert_allclose(dy_test, dy_expected)

    def test_pos_alt(self):
        """ Merging of multiple pos using the short notations. """
        count_row = 4

        pos_arr = ((Pos(50, 50) / Pos(50, 50)) + Pos(50, 50)) / Pos(100, 50)

        len_test = len(pos_arr)
        len_expected = count_row
        self.assertEqual(len_test, len_expected)

        pos_expected = [50, 50, 100, 50 * 3 / 2]
        pos_test = [p.dy for p in pos_arr]
        assert_allclose(pos_test, pos_expected)

        pos_expected = [50, 50, 100, 100 * 3 / 2]
        pos_test = [p.dx for p in pos_arr]
        assert_allclose(pos_test, pos_expected)


class TestPopulated(unittest.TestCase):
    """ Saving metadata and images into the structure. """

    def test_simple_merge_paths(self):
        """ Test the merging of items with paths """
        n_cols = 3
        paths_expected = [Path(f"/tmp/img-{i}.png") for i in range(n_cols)]
        pos_arr = create_pos_array_opts(x_size=40, y_num=n_cols, paths=paths_expected)

        paths_test = [p.path for p in pos_arr]
        self.assertEqual(paths_test, paths_expected)


if __name__ == "__main__":
    unittest.main()
