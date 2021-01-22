#!/usr/bin/env python3

""" Handle the resizing and loading of images. """

from skimage import io, transform
from pathlib import Path
import numpy as np

ImData = np.ndarray


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


class ImageBlank(Image):
    """ Class for images that cannot be loaded. """

    def __init__(self, path: Path, x_size: int, y_size: int):
        self.path = path
        self.x_size: int = x_size
        self.y_size: int = y_size
        self.data: ImData = np.ones((y_size, x_size))

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
        self.data = np.ones((self.y_size, self.x_size))
