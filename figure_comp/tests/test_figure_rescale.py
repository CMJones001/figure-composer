#!/usr/bin/env python3

import itertools
import unittest
from pathlib import Path
from typing import List

import numpy as np

import figure_comp.figure_rescale as fr


class Test_Padding(unittest.TestCase):
    def test_simple_padding(self):
        """ Padding notation is somewhat confusing, test this works. """
        starting_array = np.ones([5, 5, 4])
        array_shape_expected = (7, 7, 4)

        padding_array = ((0, 2), (0, 2), (0, 0))
        test_array = np.pad(starting_array, padding_array)
        array_shape_test = test_array.shape

        self.assertEqual(array_shape_test, array_shape_expected)

        # Test that the upper left of the array is unchange
        np.testing.assert_allclose(starting_array, test_array[:5, :5])
        # And the remaining values are zero
        test_array[:5, :5] = 0
        np.testing.assert_allclose(0, test_array)


class TestRowMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load the test images. """
        image_paths = _get_test_ims()
        cls.images = fr.load_images(image_paths)
        cls.save = False

    def test_pad_addition(self):
        """ Test adding of multiple images in a row. """
        sub_images = self.images[:3]
        merged_fig = fr.merge_row_pad(sub_images, pad_mode="edge")
        if self.save:
            merged_fig.save()

        total_width = sum([i.x for i in sub_images])
        total_height = max([i.y for i in sub_images])
        shape_expected = (total_height, total_width, 4)

        shape_test = merged_fig.shape
        self.assertEqual(shape_test, shape_expected)

    def test_scale_addition(self):
        """ Test adding of multiple images in a row. """
        sub_images = self.images[:8]
        y_size = 1000

        merged_fig = fr.merge_row_scale(sub_images, y_size=y_size)
        if True:
            merged_fig.save("/tmp/test-rescale-row.png")

        total_width = sum([int(i.aspect * y_size) for i in sub_images])
        total_height = y_size
        shape_expected = (total_height, total_width, 4)

        shape_test = merged_fig.shape
        np.testing.assert_allclose(shape_test, shape_expected)


class TestColMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load the test images. """
        image_paths = _get_test_ims()
        cls.images = fr.load_images(image_paths)
        cls.save = False

    def test_pad_addition(self):
        """ Test adding of multiple images in a column. """
        sub_images = self.images[:3]
        merged_fig = fr.merge_col_pad(sub_images, pad_mode="edge")
        if self.save:
            merged_fig.save()

        total_width = max([i.x for i in sub_images])
        total_height = sum([i.y for i in sub_images])
        shape_expected = (total_height, total_width, 4)

        shape_test = merged_fig.shape
        self.assertEqual(shape_test, shape_expected)

    def test_scale_addition(self):
        """ Test adding of multiple images in a column, scaling to the biggest image. """
        sub_images = self.images[:8]
        x_size = 1000

        merged_fig = fr.merge_col_scale(sub_images, x_size=x_size)
        if True:
            merged_fig.save("/tmp/test-rescale-col.png")

        total_width = x_size
        total_height = sum([int(x_size / i.aspect) for i in sub_images])
        shape_expected = (total_height, total_width, 4)

        shape_test = merged_fig.shape
        np.testing.assert_allclose(shape_test, shape_expected)


class TestDualMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load the test images. """
        image_paths = _get_test_ims()
        cls.images = fr.load_images(image_paths)
        cls.save = True

    def test_pad_addition(self):
        """ Test adding of two rows into a grid. """
        sub_images = self.images[:4]

        first_row = fr.merge_row_pad(self.images[:2])
        second_row = fr.merge_row_pad(self.images[2:4])

        merged_fig = fr.merge_col_pad([first_row, second_row], pad_mode="edge")
        if self.save:
            merged_fig.save("/tmp/test-dual-col-pad.png")

        total_width = max([i.x for i in [first_row, second_row]])
        total_height = sum([i.y for i in [first_row, second_row]])
        shape_expected = (total_height, total_width, 4)

        shape_test = merged_fig.shape
        self.assertEqual(shape_test, shape_expected)

    def test_scale_addition(self):
        """ Test adding of two rows into a grid. """
        sub_images = self.images[:7]

        y_size = 500
        first_row = fr.merge_row_scale(self.images[:5], y_size=y_size)
        second_row = fr.merge_row_scale(self.images[5:7], y_size=y_size)

        x_size = 2500
        total_width = x_size
        total_height = sum([round(x_size / i.aspect) for i in [first_row, second_row]])

        merged_fig = fr.merge_col_scale([first_row, second_row], x_size=x_size)
        if True:
            merged_fig.save("/tmp/test-dual-col-scale.png")

        shape_expected = (total_height, total_width, 4)

        shape_test = merged_fig.shape
        self.assertEqual(shape_test, shape_expected)

    def test_short_addition(self):
        """ Test adding of two rows into a grid in a single command. """
        sub_images = self.images

        y_size = 500
        x_size = 2500
        merged_fig = fr.merge_col_scale(
            [
                fr.merge_row_scale([self.images[1], self.images[2]], y_size=y_size),
                self.images[3],
            ],
            x_size=x_size,
        )

        total_width = x_size
        total_height = sum([round(x_size / i.aspect) for i in sub_images[:3]]) / 2

        if True:
            merged_fig.save("/tmp/test-dual-col-scale-short.png")

        shape_expected = (total_height, total_width, 4)

        shape_test = merged_fig.shape
        self.assertEqual(shape_test, shape_expected)


def _get_test_ims() -> List[Path]:
    """ Get a sample of images from the test directories. """
    project_dir = Path(__file__).resolve().parents[1]
    test_im_dir = project_dir / "tests/test_im/"

    square_ims = (test_im_dir / "square").glob("*.png")
    rectangle_ims = (test_im_dir / "wide").glob("*.png")

    return itertools.chain(square_ims, rectangle_ims)


if __name__ == "__main__":
    unittest.main()
