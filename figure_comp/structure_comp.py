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
import figure_comp.coordinate_tracking as ct


@dataclass
class _Container:
    cont: List[Union["_Container", ct.Pos]]

    def __getitem__(self, k):
        """ Indexing the _Containter indexes the cont."""
        if isinstance(k, int):
            return self.cont[k]
        raise TypeError("Index for _Container must be an integer")

    def run(self) -> ct.PosArray:
        def activate(c: List[Union["_Container", ct.Pos, ct.PosArray]]):
            """ Resolve the nested containers or pass images through """
            return c.run() if isinstance(c, _Container) else c

        cont = [activate(c) for c in self.cont]
        return self.merge_func(cont, **self.args)

    def outline(self):
        def print_row(container, offset=4):
            for item in container:
                if isinstance(item, _Container):
                    print_row(item, offset=offset + 4)
                else:
                    print(f"{' '*offset}{item}")

        print_row(self.cont)


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
