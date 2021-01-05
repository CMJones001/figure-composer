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

from figure_comp.figure_rescale import Image
import figure_comp.figure_rescale as fr
from dataclasses import dataclass
from typing import Union, List


@dataclass
class _Container:
    cont: List[Union["_Container", Image]]

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
        self.args = dict(y_size=self.y_size)
        self.merge_func = fr.merge_row_scale

    def cmd(self):
        args = ", ".join([f"{k}={v}" for k, v in self.args.items()])
        cmd = f"merge_row_{self.scale}({self.cont}, {args})"
        return cmd


@dataclass
class Col(_Container):
    x_size: int = 500

    def __post_init__(self):
        self.scale = "scale"
        self.args = dict(x_size=self.x_size)
        self.merge_func = fr.merge_col_scale

    def cmd(self):
        args = ", ".join([f"{k}={v}" for k, v in self.args.items()])
        cmd = f"merge_col_{self.scale}({self.cont}, {args})"
        return cmd


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
