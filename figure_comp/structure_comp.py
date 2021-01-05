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


@dataclass
class Row(_Container):
    y_size: int = 500

    def __post_init__(self):
        self.scale = "scale"
        self.args = dict(y_size=self.y_size)

    def cmd(self):
        args = ", ".join([f"{k}={v}" for k, v in self.args.items()])
        cmd = f"merge_row_{self.scale}({self.cont}, {args})"
        return cmd

    def run(self) -> fr.MergedImage:
        def activate(c: List[Union["_Container", Image]]):
            """ Resolve the nested containers or pass images through """
            return c.cmd() if isinstance(c, _Container) else c

        cont = [activate(c) for c in self.cont]
        # cont = self.cont
        return fr.merge_row_scale(cont, **self.args)


if __name__ == "__main__":
    import numpy as np

    test_data = [np.ones([3, 3, 3]) * 24 * i for i in range(10)]
    test_images = [fr.Image(t, path=None) for t in test_data]

    row = Row(test_images, y_size=100)
    merged_im = row.run()

    print(merged_im.shape)
    merged_im.save("/tmp/test-merged-row.png")
