#!/usr/bin/env python

"""" Attempt to merge figures by manual image rescaling.

We have multiple means to resize images:
- scipiy.ndimage.zoom
- skimage.transform.resize

Maybe look into hydra here?

Image conventions
-----------------
The image is described by array with dimensions (Y, X, 4)
where Y is the number of rows (Y-dimension) and X the number of columns.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from icecream import ic
from skimage import io, transform
from PIL import ImageFont, Image, ImageDraw
from dataclasses import dataclass, field

Im = np.ndarray

PROJECT_DIR = Path(__file__).resolve().parents[1]
FONT = PROJECT_DIR / "fonts/cm-unicode-0.7.0/cmunrm.ttf"


@dataclass
class Label:
    text: str
    pos: (int, int)
    colour: Tuple = field(default_factory=lambda: (0, 0, 0))


def merge_image_row(images: List[Im], y_size: int):
    """Combine multiple images, while resizing them all to match the image in
    index ``fit_to_image``

    Notes
    -----
    We might also calculate the fill width of the figure, then calculate this as
    a fraction of the desired width.
    """
    resized_images = [resize_skimage(i, new_y=y_size) for i in images]
    ic([im.shape for im in resized_images])
    merged_images = np.concatenate(resized_images, axis=1)

    return merged_images


def merge_image_column(images: List[Im], x_size: int):
    resized_images = [resize_skimage(i, new_x=x_size) for i in images]
    merged_images = np.concatenate(resized_images, axis=0)
    return merged_images


def resize_skimage(
    image: Im, new_x: Optional[int] = None, new_y: Optional[int] = None, order=3
):
    """Image resizing without a blurring pass

    As we want to preserve aspect ratio, we might instead change the signature to
    (new_x=None, new_y=None) and only allow one of these to be not None.
    """
    new_dims, _ = _get_new_dimensions(image.shape, new_x=new_x, new_y=new_y)
    return transform.resize(image, new_dims, order=order, preserve_range=2)


def resize_pyramid(
    image: Im, new_x: Optional[int] = None, new_y: Optional[int] = None, order: int = 3,
):
    """ """
    new_dims, rescale_factor = _get_new_dimensions(
        image.shape, new_x=new_x, new_y=new_y
    )
    return transform.pyramid_expand(
        image, rescale_factor, multichannel=True, preserve_range=True, order=order,
    )


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


def join_along_y(images: List[Im]):
    """ Join the arrays along the y-axis. """
    y_dims = [i.shape[0] for i in images]
    if any(y_dims != images[0].shape[0]):
        raise ValueError("All Y dimensions (axis 0) must be of the same size.")


def load_images(figure_paths: List[Path]) -> List[np.ndarray]:
    """ Load the path of images into an array. """
    images = [io.imread(figure_path) for figure_path in figure_paths]
    if len(images) == 0:
        raise FileNotFoundError("No images found")
    return images


def annotate_axis(image: Im, labels: List[Label]):
    """ Add text labels to the axis. """
    # The layout engine arg is required to fix a segfault
    font = ImageFont.truetype(str(FONT), 50, layout_engine=ImageFont.LAYOUT_BASIC)
    pil_image = Image.fromarray(image)
    pil_editable = ImageDraw.Draw(pil_image)
    for label in labels:
        pil_editable.text(label.pos, label.text, label.colour, font=font)
    arr = np.array(pil_image)
    return arr


def _get_test_ims() -> List[Path]:
    """ Get a sample of images from the test directories. """
    project_dir = Path(__file__).resolve().parent
    test_im_dir = project_dir / "tests/test_im/"

    square_ims = (test_im_dir / "square").glob("*.png")
    rectangle_ims = (test_im_dir / "wide").glob("*.png")

    square_im = next(square_ims)
    rectangle_im = next(rectangle_ims)
    return [square_im, rectangle_im]


def save_image(save_path: Path, image: Im):
    """ Save an image to disk"""
    io.imsave(save_path, image.astype(np.uint8), check_contrast=False)


if __name__ == "__main__":
    im_paths = _get_test_ims()
    images = load_images(im_paths)[0]

    labels = [f"{i}." for i in "abcd"]
    pos = [(10 + 50 * i, 20) for i in range(len(labels))]
    labels = [Label(l, p) for l, p in zip(labels, pos)]
    annotated = annotate_axis(images, labels)
    save_image("/tmp/annotated.png", annotated)

    ic(pos)

    # merged_im = merge_image_row(images, y_size=1000)
    # merged_im = merge_image_column(images, x_size=1000)
    # ic(merged_im)
    # save_image("/tmp/merged.png", merged_im)
