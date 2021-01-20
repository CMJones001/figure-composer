#!/usr/bin/env python3

"""" Parse the figure strutre.

Outline
-------
The idea for the figure layout should be given as a series of nested dictionaries?

Examples
++++++++

Row([Im1, Im2, Im3], mode="scale") -> Three images in a row

{type: Col, cont=[{type: Row, cont=[Im1, Im2]}, [Im3]}} -> Two images stacked on a third
A B
-C-

This needs to be converted to some set of commands

"""

from dataclasses import dataclass
from typing import List, Union

import figure_comp.figure_rescale as fr
from figure_comp.figure_rescale import Image
import figure_comp.coordinate_tracking as ct


@dataclass
class _Container:
    cont: List[Union["_Container", Image]]

    def __getitem__(self, k):
        """ Indexing the _Containter indexes the cont."""
        if isinstance(k, int):
            return self.cont[k]
        raise TypeError("Index for _Container must be an integer")

    def run(self) -> fr.MergedImage:
        def activate(c: List[Union["_Container", Image]]):
            """ Resolve the nested containers or pass images through """
            return c.run() if isinstance(c, _Container) else c

        cont = [activate(c) for c in self.cont]
        return self.merge_func(cont, **self.args)


@dataclass
class Row(_Container):
    y_size: int = 500

    def __post_init__(self):
        self.scale = "scale"
        self.args = dict()
        self.merge_func = ct.merge_row


@dataclass
class Col(_Container):
    x_size: int = 500

    def __post_init__(self):
        self.scale = "scale"
        self.args = dict()
        # self.merge_func = fr.merge_col_scale
        self.merge_func = ct.merge_col


if __name__ == "__main__":
    import numpy as np

    n_repeats = 10
    block_height = 100
    test_data = [np.ones([3, 3, 3]) * i * 200 // n_repeats for i in range(n_repeats)]
    test_images = [fr.Image(t, path=None) for t in test_data]

    test_images.append(
        Row([fr.Image(np.zeros([3, 3, 3]), path=None)], y_size=block_height),
    )

    row = Row(test_images, y_size=block_height)
    merged_im = row.run()

    shape_test = merged_im.shape
    shape_expected = (block_height, block_height * n_repeats, 3)
