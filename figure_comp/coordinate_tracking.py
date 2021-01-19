#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figure_comp.plot_tools as pt
from icecream import ic


@dataclass
class Pos:
    x: float
    y: float
    dx: float
    dy: float

    def __post_init__(self):
        if self.dx < 0 or self.dy < 0:
            raise ValueError("Pos figure widths must be positive")


@dataclass
class PosArray:
    arr: Tuple = field(default_factory=lambda: [])

    def append(self, other):
        if not isinstance(other, (PosArray)):
            raise TypeError("Unable to append non Pos type")
        self.arr = self.arr + other.arr
        return self

    def shift_x(self, x_move):
        """ Move all items right by the given amount. """
        for p in self.arr:
            p.x += x_move

    def shift_y(self, y_move):
        """ Move all items down by the given amount. """
        for p in self.arr:
            p.y += y_move

    @property
    def x_min(self):
        """ The smallest x_coordinate in the array. """
        return min(map(lambda p: p.x, self.arr))

    @property
    def x_max(self):
        """ The smallest x_coordinate in the array. """
        return max(map(lambda p: p.x + p.dx, self.arr))

    @property
    def y_min(self):
        """ The smallest y_coordinate in the array. """
        return min(map(lambda p: p.y, self.arr))

    @property
    def y_max(self):
        """ The smallest y_coordinate in the array. """
        return max(map(lambda p: p.y + p.dy, self.arr))

    def stack_right(self, other: "PosArray") -> "PosArray":
        """Add a PosArray to the right of current.

        This scales the second PosArray to fit the current.
        """
        x_offset = self.x_max - other.x_min

        y_range_self = self.y_max - self.y_min
        y_range_other = other.y_max - other.y_min
        scale_factor = y_range_self / y_range_other

        other.shift_x(x_offset)
        other.rescale(scale_factor)
        return self.append(other)

    def stack_below(self, other: "PosArray") -> "PosArray":
        """Add a PosArray below the current.

        This scales the second PosArray to fit the current.
        """
        y_offset = self.y_max - other.y_min

        x_range_self = self.x_max - self.x_min
        x_range_other = other.x_max - other.x_min
        scale_factor = x_range_self / x_range_other

        other.shift_y(y_offset)
        other.rescale(scale_factor)
        return self.append(other)

    def rescale(self, scale, about: Optional[float] = None):
        """Rescale all images by a factor.

        The starting points are rescaled as well, about a given point, typically about the
        minimium of the x positions of the included images.
        """
        min_x = min(map(lambda p: p.x, self.arr))
        min_y = min(map(lambda p: p.y, self.arr))

        for p in self.arr:
            from_scale_point_x = p.x - min_x
            from_scale_point_y = p.y - min_y

            p.x = min_x + from_scale_point_x * scale
            p.y = min_y + from_scale_point_y * scale

            p.dx = p.dx * scale
            p.dy = p.dy * scale

    def __add__(self, other):
        """If given a PosArray then stack on the right, otherwise shift by
        (x, y) coordinates.
        """
        if isinstance(other, PosArray):
            return self.stack_right(other)
        x, y = other
        self.shift_x(x)
        self.shift_y(y)
        return self

    def __mul__(self, scale):
        self.rescale(scale)
        return self

    def __truediv__(self, other):
        """ Shorthand for stack below. """
        if not isinstance(other, PosArray):
            raise TypeError(
                f"Division not supported between PosArray and {type(other)}"
            )
        return self.stack_below(other)

    def __len__(self):
        return self.arr.__len__()

    def __getitem__(self, index):
        """ Pass indexing to the inner array. """
        return self.arr[index]

    def sketch(self, save_path="/tmp/outline-sketch.png", label=False):
        """ Plot the sizes in the position array. """
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        aspect = x_range / y_range
        fig, ax = pt.create_axes(1, aspect=aspect, fig_width=6)

        for num, p in enumerate(self):
            ax.add_patch(
                Rectangle([p.x, p.y], p.dx, p.dy, lw=4, ec="black", fc="lightgrey")
            )
            if label:
                pos = (p.x + p.dx / 2, p.y + p.dy / 2)
                ax.annotate(f"{chr(num+0x41)}", pos, ha="center", va="center")

        pad = 0.1
        ax.set_xlim(self.x_min - pad * x_range, self.x_max + pad * x_range)
        ax.set_ylim(self.y_min - pad * y_range, self.y_max + pad * y_range)
        ax.invert_yaxis()
        ax.set_aspect("equal", "box")
        pt.save(save_path)
