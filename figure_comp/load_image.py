#!/usr/bin/env python3

""" Handle the resizing and loading of images. """

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import PIL
from PIL import ImageDraw, ImageFont
from skimage import io, transform
from typing import List, Optional, Tuple
from icecream import ic

ImData = np.ndarray

# TODO: Make font choice dynamic
PROJECT_DIR = Path(__file__).resolve().parents[1]
FONT = PROJECT_DIR / "fonts/cm-unicode-0.7.0/cmunrm.ttf"


@dataclass
class Label:
    """Storage for label properties.

    The position is given in relative coordinates in the range [0, 1].
    """

    text: str
    pos: (float, float)
    colour: Tuple = field(default_factory=lambda: (0, 0, 0))

    def __post_init__(self):
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
        # The layout engine arg is required to fix a segfault
        relative_pos = (np.array(label.pos) * self.data.shape[:2]).astype(np.int)

        font = ImageFont.truetype(str(FONT), 50, layout_engine=ImageFont.LAYOUT_BASIC)
        pil_image = PIL.Image.fromarray(self.data.astype(np.uint8))
        pil_editable = ImageDraw.Draw(pil_image)
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


def generate_default_label_text(format_str: None):
    """ Generator for default labels. """
    if format_str is None:
        format_str = "{index+1}."
    index = 0
    while True:
        # Evaulate the template as if it were a fstring
        label = eval(f'f"{format_str}"')
        yield label
        index += 1
