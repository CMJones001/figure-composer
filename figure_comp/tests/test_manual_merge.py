#!/usr/bin/env python3

import unittest as ut
import figure_comp.manual_merge as mm
from numpy.testing import assert_equal, assert_allclose
from icecream import ic

ic.disable()


class Test_get_new_dimensions_x(ut.TestCase):
    def test_square_growth(self):
        """ Test a doubling of the size of a square figure. """
        old_scale = (1000, 1000)
        new_y = 2000

        scale_expected = (2000, 2000), 2
        scale_test = mm._get_new_dimensions(old_scale, new_y=new_y)

        assert_equal(scale_test, scale_expected)

    def test_rect_growth(self):
        """ Test a doubling of the size of a rect. figure. """
        old_scale = (1000, 500)
        new_y = 2000

        scale_expected = (2000, 1000), 2
        scale_test = mm._get_new_dimensions(old_scale, new_y=new_y)

        assert_equal(scale_test, scale_expected)

    def test_square_shrink(self):
        """ Test a halivng of the size of a square figure. """
        old_scale = (1000, 1000)
        new_y = 100

        scale_expected = (100, 100), 0.1
        scale_test = mm._get_new_dimensions(old_scale, new_y=new_y)

        assert_equal(scale_test, scale_expected)

    def test_rect_shrink(self):
        """ Test a halivng of the size of a rect. figure. """
        old_scale = (1000, 500)
        new_y = 100

        scale_expected = (100, 50), 0.1
        scale_test = mm._get_new_dimensions(old_scale, new_y=new_y)

        assert_equal(scale_test, scale_expected)

    def test_rect_shrink_down(self):
        """Test a halving of the size of a rectangle figure.
        With rounding down of the alterative axis."""
        old_scale = (1500, 1000)
        new_y = 500

        scale_expected = (500, 333), 1.0 / 3.0
        scale_test = mm._get_new_dimensions(old_scale, new_y=new_y)

        assert_equal(scale_test, scale_expected)

    def test_rect_shrink_up(self):
        """Test a halving of the size of a rectangle figure.
        With rounding up of the alterative axis."""
        old_scale = (1500, 1000)
        new_y = 1000

        scale_expected = (1000, 667), 2.0 / 3.0
        scale_test = mm._get_new_dimensions(old_scale, new_y=new_y)

        assert_equal(scale_test, scale_expected)

    def test_addtional_axis(self):
        """ Test that additonal axes remain unchanged. """
        old_scale = (1000, 500, 4)
        new_y = 100

        scale_expected = (100, 50, 4), 0.1
        scale_test = mm._get_new_dimensions(old_scale, new_y=new_y)

        assert_equal(scale_test, scale_expected)


class Test_get_new_dimensions_y(ut.TestCase):
    def test_square_growth(self):
        """ Test a doubling of the size of a square figure. """
        old_scale = (1000, 1000)
        new_x = 2000

        scale_expected = (2000, 2000), 2
        scale_test = mm._get_new_dimensions(old_scale, new_x=new_x)

        assert_equal(scale_test, scale_expected)

    def test_rect_growth(self):
        """ Test a doubling of the size of a rect. figure. """
        old_scale = (1000, 500)
        new_x = 2000

        scale_expected = (4000, 2000), 4
        scale_test = mm._get_new_dimensions(old_scale, new_x=new_x)

        assert_equal(scale_test, scale_expected)

    def test_square_shrink(self):
        """ Test a halivng of the size of a square figure. """
        old_scale = (1000, 1000)
        new_x = 100

        scale_expected = (100, 100), 0.1
        scale_test = mm._get_new_dimensions(old_scale, new_x=new_x)

        assert_equal(scale_test, scale_expected)

    def test_rect_shrink(self):
        """ Test a halivng of the size of a rect. figure. """
        old_scale = (1000, 500)
        new_x = 100

        scale_expected = (200, 100), 0.2
        scale_test = mm._get_new_dimensions(old_scale, new_x=new_x)

        assert_equal(scale_test, scale_expected)

    def test_rect_shrink_down(self):
        """Test a halving of the size of a rectangle figure.
        With rounding down of the alterative axis."""
        old_scale = (1000, 1500)
        new_x = 500

        scale_expected = (333, 500), 1.0 / 3.0
        scale_test = mm._get_new_dimensions(old_scale, new_x=new_x)

        assert_equal(scale_test, scale_expected)

    def test_rect_shrink_up(self):
        """Test a halving of the size of a rectangle figure.
        With rounding up of the alterative axis."""
        old_scale = (1000, 1500)
        new_x = 1000

        scale_expected = (667, 1000), 2.0 / 3.0
        scale_test = mm._get_new_dimensions(old_scale, new_x=new_x)

        assert_equal(scale_test, scale_expected)


class Test_get_new_dimensions_raises(ut.TestCase):
    def test_no_new(self):
        """ At least one new dim must be given. """
        self.assertRaises(ValueError, mm._get_new_dimensions, (1000, 1000))

    def test_both_new(self):
        """ Only one new dim must be given. """
        self.assertRaises(
            ValueError, mm._get_new_dimensions, (1000, 1000), new_y=10, new_x=8
        )
