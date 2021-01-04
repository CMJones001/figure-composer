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
from typing import List, Optional, Tuple

import itertools
import numpy as np
import PIL
from icecream import ic
from PIL import ImageDraw, ImageFont
from skimage import io, transform

# TODO: Dynamically load the fonts
PROJECT_DIR = Path(__file__).resolve().parents[1]
FONT = PROJECT_DIR / "fonts/cm-unicode-0.7.0/cmunrm.ttf"


@dataclass
class Label:
    text: str
    pos: (int, int)
    colour: Tuple = field(default_factory=lambda: (0, 0, 0))


class Image:
    def __init__(self, data: np.ndarray, path: Path):
        self.data = data
        self.path = path
        self.original_data = self.data.copy()

    @property
    def shape(self):
        return self.data.shape

    @property
    def physical_size(self):
        # TODO: Import the metadata from the image.
        dpi = 300
        return self.data.shape * dpi

    def __repr__(self):
        return (
            f"Image({self.path}, new shape {self.shape}, old shape"
            f" {self.original_data.shape})"
        )

    def resize(self, new_x: Optional[int] = None, new_y: Optional[int] = None, order=3):
        """ Return a resized image. Note the original is not changed! """
        new_dims, _ = _get_new_dimensions(
            self.original_data.shape, new_x=new_x, new_y=new_y
        )
        resized = transform.resize(
            self.data, new_dims, order=order, preserve_range=True
        )
        return resized

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

    def __post_init(self):
        # TODO: Get a list of figures positions within the main body
        pass


def load_images(figure_paths: List[Path]) -> List[Image]:
    """ Load the path of images into an array. """
    images = [Image(io.imread(path), path) for path in figure_paths]
    if len(images) == 0:
        raise FileNotFoundError("No images found")
    return images


def merge_image_row(images: List[Image], y_size: int):
    """Combine multiple images, while resizing them all to match the image in
    index ``fit_to_image``

    Notes
    -----
    We might also calculate the fill width of the figure, then calculate this as
    a fraction of the desired width.
    """
    resized_images = [i.resize(new_y=y_size) for i in images]
    ic([im.shape for im in images])

    merged_data = np.concatenate(resized_images, axis=1)
    merged_image = MergedImage(merged_data, resized_images)

    return merged_image


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
    images = load_images(im_paths)[:5]

    labels = [f"{i}." for i in "abcdefghi"[:5]]
    pos = [(30 + 1000 * i, 20) for i in range(len(labels))]
    labels = [Label(l, p) for l, p in zip(labels, pos)]

    # for image, label in zip(images, labels):
    #     image.annotate([label])
    #     ic(image.shape)

    merged_im = merge_image_row(images, y_size=1000)
    ic(merged_im.shape)
    merged_im.annotate(labels)
    merged_im.save("/tmp/merged_auto.png")
    ic(images)
