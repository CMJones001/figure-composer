#!/usr/bin/env python3

""" Test the image operations. """

import unittest
import figure_comp.load_image as li
from skimage.io import imsave
from pathlib import Path

test_dir = Path(__file__).resolve().parent
test_im_dir = test_dir / "test_im"

PLOT = False


class TestAnnotate(unittest.TestCase):
    def test_annotate_blank(self):
        """ Add a label to a blank image. """
        x_size = 500
        y_size = 500

        label = li.Label("Test Label", (0.20, 0.20))
        test_im = li.ImageBlank(None, x_size, y_size)
        test_im.annotate(label)
        annotated_im = test_im.data

        # Ensure the shape is retained
        shape_expected = (y_size, x_size, 4)
        shape_test = annotated_im.shape
        self.assertEqual(
            shape_test, shape_expected, msg="shape not retained after annotation"
        )

        # Test that there are darker text areas
        flattened = annotated_im[..., :3].mean(axis=-1)
        n_black_pixels = (flattened < 10).sum()

        self.assertGreater(n_black_pixels, 50, msg="Not enough black/text pixels")
        self.assertLess(n_black_pixels, 5000, msg="Too many black/text pixels")

    def test_annotate_image(self):
        """ Annotate an image on disk. """
        im_path = test_im_dir / "square-im-1.png"
        image = li.Image(im_path)

        label = li.Label("Test Label", (0.50, 0.50))
        image.annotate(label)
        annotated_im = image.data

        # Ensure the shape is retained
        shape_expected = image.data_original.shape
        shape_test = annotated_im.shape
        self.assertEqual(
            shape_test, shape_expected, msg="shape not retained after annotation"
        )

        if PLOT:
            imsave("/tmp/annotate_image.png", annotated_im, check_contrast=False)


if __name__ == "__main__":
    unittest.main()
