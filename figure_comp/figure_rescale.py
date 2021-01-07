#!/usr/bin/env python3

"""" Introduce an image class to allow resizing and addition of images.

Image conventions
-----------------
The image is described by array with dimensions (Y, X, 4)
where Y is the number of rows (Y-dimension) and X the number of columns.

Notes
-----
It would be simpler to annotate the axes before merging, on their own coordinate system,
however, if the images are then rescaled differently
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from copy import copy
import itertools
import numpy as np
import PIL
from icecream import ic
from PIL import ImageDraw, ImageFont
from skimage import io, transform

# TODO: Dynamically load the fonts
PROJECT_DIR = Path(__file__).resolve().parents[1]
FONT = PROJECT_DIR / "fonts/cm-unicode-0.7.0/cmunrm.ttf"

Im = np.ndarray


@dataclass
class Label:
    text: str
    pos: (int, int)
    colour: Tuple = field(default_factory=lambda: (0, 0, 0))


class Image:
    def __init__(self, data: Im, path: Path, original_data: Optional[Im] = None):
        self.data = data
        self.path = Path(path)
        self.original_data = (
            self.data.copy() if original_data is None else original_data
        )

    @property
    def y(self):
        return self.data.shape[0]

    @property
    def x(self):
        return self.data.shape[1]

    @property
    def shape(self):
        return self.data.shape

    @property
    def physical_size(self):
        # TODO: Import the metadata from the image.
        dpi = 300
        return self.data.shape * dpi

    @property
    def aspect(self):
        return self.x / self.y

    def __eq__(self, o):
        if not isinstance(o, Image):
            raise TypeError("Comparing image to non-image")
        path_eq = self.path.samefile(o.path)
        data_eq = np.allclose(self.data, o.data)
        return path_eq and data_eq

    def __repr__(self):
        return (
            f"Image({self.path}, new shape {self.shape}, old shape"
            f" {self.original_data.shape})"
        )

    def pad(
        self, new_x: Optional[int] = None, new_y: Optional[int] = None, mode="constant"
    ):
        """Pad the array with zeros.

        For now, we pad left and down.
        """
        pad_x = new_x - self.x if new_x else 0
        pad_y = new_y - self.y if new_y else 0

        if pad_x < 0 or pad_y < 0:
            raise ValueError(
                "Attempting to pad array to smaller size."
                f"({self.shape}) -> ({new_y, new_x, self.shape[2]}) "
            )

        # Return in the trival case
        if pad_x == 0 and pad_y == 0:
            return self.data

        padding = ((0, pad_y), (0, pad_x), (0, 0))
        padded_im = np.pad(self.data, padding, mode=mode)

        im_copy = copy(self)
        im_copy.data = padded_im
        return im_copy

    def resize(self, new_x: Optional[int] = None, new_y: Optional[int] = None, order=3):
        """ Return a resized image. Note the original is not changed! """
        new_dims, _ = _get_new_dimensions(
            self.original_data.shape, new_x=new_x, new_y=new_y
        )
        resized = transform.resize(
            self.data, new_dims, order=order, preserve_range=True
        )

        im_copy = copy(self)
        im_copy.data = resized
        return im_copy

    def annotate(self, labels: List[Label], inplace=True):
        """ Add text labels to the image. """
        # The layout engine arg is required to fix a segfault
        font = ImageFont.truetype(str(FONT), 50, layout_engine=ImageFont.LAYOUT_BASIC)
        pil_image = PIL.Image.fromarray(self.data.astype(np.uint8))
        pil_editable = ImageDraw.Draw(pil_image)
        for label in labels:
            pil_editable.text(label.pos, label.text, label.colour, font=font)
        new_data = np.array(pil_image)
        if inplace:
            self.data = new_data
        return new_data

    def save(self, save_path: Path):
        """ Save the image to disk"""
        io.imsave(save_path, self.data.astype(np.uint8), check_contrast=False)


@dataclass
class MergedImage(Image):
    """ Container for a figure containing multiple images. """

    def __init__(self, data: np.ndarray, images: List[Image]):
        self.data = data
        self.images = images
        self.original_data = self.data.copy()


def load_images(figure_paths: List[Path]) -> List[Image]:
    """ Load the path of images into an array. """
    images = [Image(io.imread(path), path) for path in figure_paths]
    if len(images) == 0:
        raise FileNotFoundError("No images found")
    return images


def merge_row_scale(images: List[Image], y_size: Optional[int] = None):
    """Combine multiple images, while resizing them all to match the image in
    index ``fit_to_image``

    Notes
    -----
    We might also calculate the fill width of the figure, then calculate this as
    a fraction of the desired width.
    """
    if y_size is None:
        y_size = max([i.y for i in images])
    resized_images = [i.resize(new_y=y_size) for i in images]
    merged_data = np.concatenate([i.data for i in resized_images], axis=1)
    return MergedImage(merged_data, resized_images)


def merge_row_pad(images: List[Image], pad_mode="edge"):
    """ Combine images into a row, padding any hieight difference. """
    max_y = max([i.y for i in images])
    padded_images = [i.pad(new_y=max_y, mode=pad_mode) for i in images]
    merged_data = np.concatenate([i.data for i in padded_images], axis=1)
    return MergedImage(merged_data, padded_images)


def merge_col_scale(images: List[Image], x_size: Optional[int] = None):
    """ Combine images into a row, padding any hieight difference. """
    if x_size is None:
        x_size = max([i.x for i in images])
    resized_images = [i.resize(new_x=x_size) for i in images]
    merged_data = np.concatenate([i.data for i in resized_images], axis=0)
    return MergedImage(merged_data, resized_images)


def merge_col_pad(images: List[Image], pad_mode="edge"):
    """ Combine images into a row, padding any hieight difference. """
    max_x = max([i.x for i in images])
    padded_images = [i.pad(new_x=max_x, mode=pad_mode) for i in images]
    merged_data = np.concatenate([i.data for i in padded_images], axis=0)
    return MergedImage(merged_data, padded_images)


def _get_new_dimensions(
    old_dims: (int, int), new_x: Optional[int] = None, new_y: Optional[int] = None
) -> ((int, int), float):
    """Give the dimensions of the new image when rescaling, keeping the aspect ratio.

    Parameters
    ----------
    Only one of ``new_x`` and ``new_y`` may be not None. The given value will be used to
    rescale the image to match this dimension.

    Returns
    -------
    (int, int): The new dimensions of the image
    (float): The rescale factor required

    Notes
    -----
    When casting to integers, python truncates floats. Adding 0.5 before truncation acts
    as rounding.
    """
    old_dims = np.array(old_dims, dtype=int)
    if new_x is not None and new_y is not None:
        raise ValueError("Only one of ``new_x`` and ``new_y`` can be given.")
    elif new_x is not None:
        # Set from x
        rescale_factor = new_x / old_dims[1]
        new_dims = old_dims.copy()
        new_dims[0] = old_dims[0] * rescale_factor + 0.5
        new_dims[1] = new_x
        return new_dims, rescale_factor
    elif new_y is not None:
        # Set from y
        rescale_factor = new_y / old_dims[0]
        new_dims = old_dims.copy()
        new_dims[1] = old_dims[1] * rescale_factor + 0.5
        new_dims[0] = new_y
        return new_dims, rescale_factor
    else:
        raise ValueError("At least one of ``new_x`` or ``new_y`` must be given.")


def _get_test_ims() -> List[Path]:
    """ Get a sample of images from the test directories. """
    project_dir = Path(__file__).resolve().parent
    test_im_dir = project_dir / "tests/test_im/"

    square_ims = (test_im_dir / "square").glob("*.png")
    rectangle_ims = (test_im_dir / "wide").glob("*.png")

    return itertools.chain(square_ims, rectangle_ims)


if __name__ == "__main__":
    im_paths = _get_test_ims()
    images = load_images(im_paths)[4:7]

    all_ = merge_col_scale(images, x_size=1000)
    all_.save("/tmp/all_merge.png")

    # last_im = images.pop()
    # last_im = last_im.resize(new_x=2000)
    # images.append(last_im)

    # new_arr = images[0].pad(new_y=2000, mode="edge")
    # im = Image(new_arr, images[0].path)
    # im.save("/tmp/padded-edge.png")
    # print(im)

    # labels = [f"{i}." for i in "abcdefghi"[:3]]
    # pos = [(30 + 1090 * i, 20) for i in range(len(labels))]
    # labels = [Label(l, p) for l, p in zip(labels, pos)]

    # merged_im = merge_row_scale(images, y_size=1000)
    # merged_im = merge_row_pad(images, pad_mode="edge")
    # ic(merged_im.shape)
    # merged_im.annotate(labels)
    # merged_im.save("/tmp/merged_auto.png")
    # ic(images)
