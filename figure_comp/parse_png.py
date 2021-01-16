#!/usr/bin/env python3

""" Read the metadata from a .png image. """

from hachoir.parser import createParser
from hachoir.metadata import extractMetadata, register
from typing import List
from pathlib import Path
from itertools import chain
from icecream import ic

from dataclasses import dataclass

# For some reason the default values of 10_000 is less than 300 dpi
register.MAX_DPI_WIDTH = 100000
register.MAX_DPI_HEIGHT = 100000


@dataclass
class ImageMetadata:
    path: Path
    dpi: int
    comments: List[Path]


def _get_test_ims() -> List[Path]:
    """ Get a sample of images from the test directories. """
    project_dir = Path(__file__).resolve().parent
    test_im_dir = project_dir / "tests/test_im/"

    square_ims = (test_im_dir / "square").glob("*.png")
    rectangle_ims = (test_im_dir / "wide").glob("*.png")

    return chain(square_ims, rectangle_ims)


def get_im_metadata(im_path: Path):
    """ Get the user provided comments and image dpi. """
    parser = createParser(str(im_path))
    with parser:
        metadata = extractMetadata(parser)

    # If we have more than one comment, then converting to a dictionary clobbers them
    metadata_dict = metadata._Metadata__data
    comment_tree = metadata_dict["comment"].values
    comments = [c.value for c in comment_tree]

    dot_per_meter = metadata_dict["width_dpi"].values[0].value
    dpi = int(dot_per_meter / 39.3701 + 0.5)

    return ImageMetadata(im_path, dpi, comments)


if __name__ == "__main__":
    test_paths = _get_test_ims()

    for num, im_path in enumerate(test_paths):
        metadata = get_im_metadata(im_path)
        ic(metadata)
