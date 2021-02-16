#!/usr/bin/env python3

""" Handle the resizing and loading of images. """

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import PIL
from PIL import ImageDraw, ImageFont
from skimage import io, transform
from typing import List, Optional, Tuple, Generator
from icecream import ic

ImData = np.ndarray
Coordinate = Tuple[float, float]

# TODO: Make font choice dynamic
PROJECT_DIR = Path(__file__).resolve().parents[1]
FONT = PROJECT_DIR / "fonts/cm-unicode-0.7.0/cmunrm.ttf"


@dataclass
class Label:
    """Storage for label properties.

    The position is given in relative coordinates in the range [0, 1].
    """

    text: str
    pos: Coordinate
    colour: Tuple = field(default_factory=lambda: (0, 0, 0))
    size: int = 30

    def __post_init__(self):
        # Deal with position passed as a string
        if isinstance(self.pos, str):
            pos_str = self.pos.strip("()")
            self.pos = np.fromstring(pos_str, sep=", ")
        elif isinstance(self.pos, tuple):
            self.pos = np.array(self.pos)
        if self.pos.max() > 1 or self.pos.min() < 0:
            error_msg = f"Label position range must be in the range [0, 1]: {self.pos}"
            raise ValueError(error_msg)


class Image:
    def __init__(self, path: Path):
        self.path: Path = Path(path)
        self.data: ImData = self.load(path)
        self.data_original: ImData = self.data.copy()

    @staticmethod
    def load(path: Path) -> ImData:
        """ Load the image from disk. """
        return io.imread(path)

    @property
    def x(self):
        return self.data.shape[1]

    @property
    def y(self):
        return self.data.shape[0]

    def resize(self, new_size, order=3):
        """ Resize the image. """
        if len(new_size) == 3:
            pass
        if len(new_size) == 2:
            new_size = (*new_size, self.data_original.shape[-1])
        else:
            raise ValueError(f"Invalid resize shape {new_size}: must be len 2 or 3 ")

        self.data = transform.resize(
            self.data_original, new_size, order=order, preserve_range=True
        )

    def annotate(self, label):
        """ Add text labels to the image. """
        # Coordinates relative to this image, reversed as PIL and numpy have different orders
        relative_pos = (np.array(label.pos) * self.data.shape[:2]).astype(np.int)[::-1]

        # The layout engine arg is required to fix a segfault
        font = ImageFont.truetype(
            str(FONT), label.size, layout_engine=ImageFont.LAYOUT_BASIC
        )

        # Convert into the PIL type that supports annotations
        pil_image = PIL.Image.fromarray(self.data.astype(np.uint8))
        pil_editable = ImageDraw.Draw(pil_image)

        # Get the size of the text box, and correct for this
        # This first method is apparently less accurate
        # Look into textbbox: anchors for better centering
        # text_shape = np.array(pil_editable.textsize(label.text, font=font))
        text_shape = np.array(font.getsize(label.text))
        relative_pos = (relative_pos - text_shape / 2).astype(np.int)

        pil_editable.text(relative_pos, label.text, label.colour, font=font)
        self.data = np.array(pil_image)


class ImageBlank(Image):
    """ Class for images that cannot be loaded. """

    def __init__(self, path: Path, x_size: int, y_size: int):
        self.path = path
        self.x_size: int = x_size
        self.y_size: int = y_size
        # Create a white image, but zero out the alpha channel
        self.data: ImData = np.ones((y_size, x_size, 4), dtype=np.uint8) * 255

    @property
    def x(self):
        return self.x_size

    @property
    def y(self):
        return self.y_size

    def resize(self, new_size, order=3):
        """ Resize the image. """
        self.x_size = new_size[0]
        self.y_size = new_size[1]
        self.data = np.ones((self.y_size, self.x_size, 4), dtype=np.uint8) * 255


def generate_default_label_text(
    format_str: Optional[str] = None,
    pos_default: Optional[Coordinate] = None,
    size_default: Optional[int] = 30,
) -> Generator[Label, None, None]:
    """Generator for default labels.

    This yields a function for creating the new label. Just calling this function will give
    the default label, but values can be overridden by kwargs to this function.

    Parameters
    ----------

    format_str: str {default: "{index+1}."}
        Template for the default label text, see 'default label generation' section for
        more information

    pos_default: (real, real)  {default: (0.05, 0.05)}
        Default (x, y) position for the label within each sub-figure, values are in the
        range [0, 1] and correspond to the fraction of the sub-figure.

    size: int {default: 30}
        Default size of the font. Given in pixel units(?)


    Default label generation
    ------------------------

    format_str is evaluated as an f-string and used as the default label text. ``index``
    starts at zero and is incremented for each figure.

    For instance:
      "{index+1}." -> 1. 2. 3. ...
      "{chr(index+0x61)}:" -> a: b: c: ...
      "({chr(index+0x41)})" -> (A) (B) (C)

    """
    if pos_default is None:
        pos_default = np.array([0.05, 0.05])
    if format_str is None:
        format_str = "{index+1}."

    index = 0
    while True:
        # Evaulate the template as if it were a fstring
        text_default = eval(f'f"{format_str}"')

        def label_func(text=text_default, pos=pos_default, size=size_default, **kwargs):
            """Generate a Label object.

            Default values will be used from the generator unless provided.
            """
            return Label(text=text, pos=pos, size=size, **kwargs)

        yield label_func
        index += 1


def default_if_none(value, default):
    """ Return ``value`` if is not None, ``default`` otherwise. """
    return value if value is not None else default
