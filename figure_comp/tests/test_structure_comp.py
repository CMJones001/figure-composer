#!/usr/bin/env python3

import inspect
import unittest
from pathlib import Path

import numpy as np
from icecream import ic

import figure_comp.figure_rescale as fr
import figure_comp.structure_comp as sc


class TestSimpleLayout(unittest.TestCase):
    """" Examine the merging of Rows and Columns by testing the output dimensions.  """

    def test_simple_row(self):
        """ Put n square images into a simple row. """
        n_repeats = 10
        block_height = 100
        test_data = [
            np.ones([3, 3, 3]) * i * 200 // n_repeats for i in range(n_repeats)
        ]
        test_images = [fr.Image(t, path=".") for t in test_data]

        row = sc.Row(test_images, y_size=block_height)
        merged_im = row.run()

        shape_test = merged_im.shape
        shape_expected = (block_height, block_height * n_repeats, 3)
        self.assertEqual(shape_test, shape_expected)

    def test_rect_row(self):
        """ Put n rectangle images into a simple row. """
        n_repeats = 10
        block_height = 100
        aspect = 2

        test_data = [
            np.ones([3, 3 * aspect, 3]) * i * 200 // n_repeats for i in range(n_repeats)
        ]
        test_images = [fr.Image(t, path=".") for t in test_data]

        row = sc.Row(test_images, y_size=block_height)
        merged_im = row.run()

        shape_test = merged_im.shape
        shape_expected = (block_height, block_height * n_repeats * aspect, 3)
        self.assertEqual(shape_test, shape_expected)

        save = True
        if save:
            merged_im.save("/tmp/test-rectangle-merge.png")

    def test_dual_row(self):
        """ Put n square images into a nested row. """
        n_repeats = 10
        block_height = 100
        test_data = [
            np.ones([3, 3, 3]) * i * 200 // n_repeats for i in range(n_repeats)
        ]
        test_images = [fr.Image(t, path=".") for t in test_data]

        test_images.append(
            sc.Row([fr.Image(np.zeros([3, 3, 3]), path=".")], y_size=block_height),
        )

        row = sc.Row(test_images, y_size=block_height)
        merged_im = row.run()

        shape_test = merged_im.shape
        shape_expected = (block_height, block_height * (n_repeats + 1), 3)
        self.assertEqual(shape_test, shape_expected)

    def test_simple_col(self):
        """ Put n square images into a simple col. """
        n_repeats = 10
        block_width = 100
        test_data = [
            np.ones([3, 3, 3]) * i * 200 // n_repeats for i in range(n_repeats)
        ]
        test_images = [fr.Image(t, path=".") for t in test_data]

        col = sc.Col(test_images, x_size=block_width)
        merged_im = col.run()

        shape_test = merged_im.shape
        shape_expected = (block_width * n_repeats, block_width, 3)
        self.assertEqual(shape_test, shape_expected)

    def test_rect_col(self):
        """ Put n rectangle images into a simple col. """
        n_repeats = 10
        block_width = 100
        aspect = 2

        test_data = [
            np.ones([3, 3 * aspect, 3]) * i * 200 // n_repeats for i in range(n_repeats)
        ]
        test_images = [fr.Image(t, path=".") for t in test_data]

        col = sc.Col(test_images, x_size=block_width)
        merged_im = col.run()

        shape_test = merged_im.shape
        shape_expected = (block_width * n_repeats / aspect, block_width, 3)
        self.assertEqual(shape_test, shape_expected)

        save = True
        if save:
            merged_im.save("/tmp/test-rectangle-col-merge.png")

    def test_dual_plot(self):
        """ Create a column and merge this to a row . """

        n_col = 3
        n_row = 2
        block_height = 200

        # Create images arrays
        row_data = [np.ones([3, 3, 3]) * i * 200 // n_col for i in range(n_col)]
        col_data = [np.ones([3, 3, 3]) * i * 100 for i in range(n_row)]

        # Turn these into an ``Image``
        row_im = [fr.Image(t, path=".") for t in row_data]
        col_im = [fr.Image(t, path=".") for t in col_data]

        # Merge the images into structures
        col = sc.Col(col_im, x_size=block_height)
        row = sc.Row([*row_im, col], y_size=block_height)

        merged_im = row.run()

        save = True
        if save:
            merged_im.save(get_save_name())

        shape_test = merged_im.shape
        shape_expected = (block_height, block_height * (n_col + 1 / n_row), 3)
        np.testing.assert_allclose(shape_test, shape_expected, atol=3)

    def test_tri_plot(self):
        """ Merge the dual plot again. """

        n_col = 3
        n_row = 2
        n_out_col = 2
        block_height = 300
        outer_width = block_height * 4

        # Create images arrays
        col_data = [np.ones([3, 3, 3]) * i * 100 for i in range(n_row)]
        row_data = [np.ones([3, 3, 3]) * i * 250 // n_col for i in range(n_col)]

        out_col_data = [np.ones([3, 3, 3]) * i * 250 // n_col for i in range(n_out_col)]

        # Turn these into an ``Image``
        col_im = [fr.Image(t, path=".") for t in col_data]
        row_im = [fr.Image(t, path=".") for t in row_data]
        out_col_im = [fr.Image(t, path=".") for t in out_col_data]

        # Merge the images into structures
        col = sc.Col(col_im, x_size=block_height)
        row = sc.Row([*row_im, col], y_size=block_height)
        outer_col = sc.Col([row, *out_col_im], x_size=outer_width)

        merged_im = outer_col.run()

        save = True
        if save:
            merged_im.save(get_save_name())

        # Properties of the first merged row
        col_height = block_height / n_row
        row_width = (n_col * block_height) + col_height
        row_height = block_height
        row_aspect = row_width / row_height

        # After merging
        out_col_width = outer_width
        out_col_height = outer_width * (n_out_col + 1 / row_aspect)

        ic(row_width)
        ic(row.run().shape)

        shape_test = merged_im.shape
        shape_expected = (out_col_height, out_col_width, 3)
        np.testing.assert_allclose(shape_test, shape_expected, atol=3)

    def test_double_merge(self):
        """ Merge two rows with a column. """

        n_col = 3
        n_row = 1
        block_height = 300

        aspect_one = 2
        aspect_two = 1

        # Create images arrays
        row_one_data = [
            np.ones([3, 3 * aspect_one, 3]) * i * 250 // n_col for i in range(n_col)
        ]
        row_two_data = [
            np.ones([3, 3 * aspect_two, 3]) * i * 250 // n_col
            for i in range(n_col, 0, -1)
        ]

        col_data = [np.ones([3, 3, 3]) * i * 250 // n_row for i in range(n_row)]

        # Turn these into an ``Image``
        row_one_im = [fr.Image(t, path=".") for t in row_one_data]
        row_two_im = [fr.Image(t, path=".") for t in row_two_data]
        col_im = [fr.Image(t, path=".") for t in col_data]

        # Merge the images into structures
        row_one = sc.Row(row_one_im, y_size=500)
        row_two = sc.Row(row_two_im, y_size=500)
        col = sc.Col([row_one, *col_im, row_two], x_size=block_height)

        merged_im = col.run()

        save = True
        if save:
            merged_im.save(get_save_name())

        row_one_height = block_height / aspect_one / n_col
        row_two_height = block_height / aspect_two / n_col
        total_height = block_height * n_row + row_one_height + row_two_height

        shape_test = merged_im.shape
        shape_expected = (total_height, block_height, 3)
        np.testing.assert_allclose(shape_test, shape_expected, atol=3)


def get_save_name(prefix=""):
    """ Create a save name from the function name. """
    save_name = inspect.stack()[1].function
    if prefix:
        save_name = prefix + save_name
    save_name += ".png"

    save_path = Path("/tmp/") / save_name
    return save_path


if __name__ == "__main__":
    unittest.main()
