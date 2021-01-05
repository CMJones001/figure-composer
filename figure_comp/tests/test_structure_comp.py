#!/usr/bin/env python3

import unittest
import figure_comp.structure_comp as sc
import figure_comp.figure_rescale as fr
from icecream import ic
import numpy as np


class TestSimpleLayout(unittest.TestCase):
    def test_simple_row(self):
        """ Put n square images into a simple row. """
        n_repeats = 10
        block_height = 100
        test_data = [
            np.ones([3, 3, 3]) * i * 200 // n_repeats for i in range(n_repeats)
        ]
        test_images = [fr.Image(t, path=None) for t in test_data]

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
        test_images = [fr.Image(t, path=None) for t in test_data]

        row = sc.Row(test_images, y_size=block_height)
        merged_im = row.run()

        shape_test = merged_im.shape
        shape_expected = (block_height, block_height * n_repeats * aspect, 3)
        self.assertEqual(shape_test, shape_expected)


if __name__ == "__main__":
    unittest.main()
