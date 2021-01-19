#!/usr/bin/env python3

import itertools
import unittest

import numpy as np
from figure_comp.coordinate_tracking import Pos, PosArray
from figure_comp.figure_rescale import Image
from figure_comp.structure_comp import Col, Row
from icecream import ic
from numpy.testing import assert_allclose


def create_row_array(x_size=50, num=5, x_offset=0, y_offset=0) -> PosArray:
    """ Create a simple row of images that are ``x_size`` wide. """
    x_pos = np.arange(0, num) * x_size + x_offset
    pos_arr = PosArray([Pos(x, y_offset, x_size, x_size) for x in x_pos])
    return pos_arr


def create_pos_array(
    x_size=50, y_size=50, x_num=1, y_num=1, x_offset=0, y_offset=0
) -> PosArray:
    """ General creation of PosArray. """
    x_pos = np.arange(0, x_num) * x_size + x_offset
    y_pos = np.arange(0, y_num) * y_size + y_offset
    pos_arr = PosArray(
        [Pos(x, y, x_size, x_size) for x, y in itertools.product(x_pos, y_pos)]
    )
    return pos_arr


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


class TestSimpleMerges(unittest.TestCase):
    """ Test that coordinates are correctly recorded across simple merges."""

    def create_row_array(self, x_size=50, num=5, x_offset=0, y_offset=0) -> PosArray:
        """ Create a simple row images that are ``x_size`` wide. """
        x_pos = np.arange(0, num) * x_size + x_offset
        pos_arr = PosArray([Pos(x, y_offset, x_size, x_size) for x in x_pos])
        return pos_arr

    def test_scaling_offset(self):
        """ Test the merging of a simple shape. """
        scale_factor = 0.5
        x_size = 50
        x_offset = 30

        count = 4
        first_row = create_row_array(x_size=x_size, num=count)

        count = 2
        second_row = create_row_array(x_size=x_size, num=count)
        second_row += (0, 50)
        second_row *= 2

        pos_arr = first_row.append(second_row)

        return

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
        """ Check adding of two coulmn shapes. """
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
        return

    def test_col_three_add(self):
        """ Check adding of two coulmn shapes and a row. """
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

        return

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
        self.assertEqual(x_max_test, x_max_expected)

        y_max_expected = 50 * (count_left)
        y_max_test = pos_arr.y_max
        self.assertEqual(y_max_test, y_max_expected)

        return

    def test_row_two_add(self):
        """ Check adding of two row shapes. """
        count_top = 2
        row_top = create_pos_array(x_num=count_top)

        count_bottom = 3
        row_bottom = create_pos_array(x_num=count_bottom)

        pos_arr = row_top.stack_below(row_bottom)

        x_max_expected = 50 * (count_top)
        x_max_test = pos_arr.x_max
        self.assertEqual(x_max_test, x_max_expected)

        y_max_expected = 50 * (1 + count_top / count_bottom)
        y_max_test = pos_arr.y_max
        self.assertEqual(y_max_test, y_max_expected)
        return

    def test_row_two_add_alt(self):
        """ Check adding of two row shapes. """
        count_top = 2
        row_top = create_pos_array(x_num=count_top)

        count_bottom = 3
        row_bottom = create_pos_array(x_num=count_bottom)

        pos_arr = row_top / row_bottom
        pos_arr.sketch(label=True)

        x_max_expected = 50 * (count_top)
        x_max_test = pos_arr.x_max
        self.assertEqual(x_max_test, x_max_expected)

        y_max_expected = 50 * (1 + count_top / count_bottom)
        y_max_test = pos_arr.y_max
        self.assertEqual(y_max_test, y_max_expected)
        return


if __name__ == "__main__":
    unittest.main()
