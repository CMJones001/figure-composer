#!/usr/bin/env python3

from functools import reduce
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from icecream import ic
from matplotlib.patches import Rectangle
from skimage import io

import figure_comp.plot_tools as pt
from figure_comp.load_image import Image, ImageBlank, Label


class Pos:
    """ Storage for the position and size of the images in figure layout. """

    def __init__(
        self,
        path: Path,
        label: Optional[Label] = None,
        dx: Optional[int] = 50,
        dy: Optional[int] = 50,
        x: Optional[int] = 0.0,
        y: Optional[int] = 0.0,
        options=None,
    ):
        """
        Parameters
        ----------

        path: Path
           Path to the image to include in the figure
        label: Label

        If the path cannot be resolved into a file, then an ImageBlank will be loaded.

        Debugging Parameters
        --------------------
        These parameters do not need be provided by the user, except for testing.

        dx, dy: Optional[int]
           The size of the image.
        x, y: Optional[int]
           The position of the upper right corner of the image.

        """
        self.dx = dx
        self.dy = dy
        self.x = x
        self.y = y
        self.options = options if path is not None else dict()
        self.label = label
        self.template = True

        # TODO: Tidy up this path resolving
        if path is None:
            self.path = Path(".")
            self.image = ImageBlank(self.path, dx, dy)
        else:
            self.path = Path(path)
            if self.path.exists():
                self.image = Image(self.path)
                self.dx = self.image.x
                self.dy = self.image.y
                self.template = False
            else:
                self.image = ImageBlank(self.path, dx, dy)

    def append(self, other):
        if not isinstance(other, (Pos, PosArray)):
            raise TypeError("Unable to append non Pos type")
        return PosArray([self, other])

    def __post_init__(self):
        if self.dx < 0 or self.dy < 0:
            raise ValueError("Pos figure widths must be positive")

    def shift_x(self, x_move):
        """ Move all items right by the given amount. """
        self.x += x_move

    def shift_y(self, y_move):
        """ Move all items down by the given amount. """
        self.y += y_move

    @property
    def x_min(self):
        """ The smallest x_coordinate in the array. """
        return self.x

    @property
    def x_max(self):
        """ The smallest x_coordinate in the array. """
        return self.x + self.dx

    @property
    def x_range(self):
        return self.x_max - self.x_min

    @property
    def y_min(self):
        """ The smallest y_coordinate in the array. """
        return self.y

    @property
    def y_max(self):
        """ The smallest y_coordinate in the array. """
        return self.y + self.dy

    @property
    def y_range(self):
        return self.y_max - self.y_min

    @property
    def aspect(self):
        return self.x_range / self.y_range

    def stack_right(self, other: "PosArray") -> "PosArray":
        """Add a PosArray to the right of current.

        This scales the second PosArray to fit the current.
        """
        x_offset = self.x_max - other.x_min
        scale_factor = self.y_range / other.y_range

        other.shift_x(x_offset)
        other.rescale(scale_factor)
        return self.append(other)

    def stack_below(self, other: "PosArray") -> "PosArray":
        """Add a PosArray below the current.

        This scales the second PosArray to fit the current.
        """
        y_offset = self.y_max - other.y_min
        scale_factor = self.x_range / other.x_range

        other.shift_y(y_offset)
        other.rescale(scale_factor)
        return self.append(other)

    def rescale(self, scale):
        """Rescale all images by a factor.

        The starting points are uneffected
        """
        self.dx = self.dx * scale
        self.dy = self.dy * scale

    def annotate(self):
        """ Add a label to the image, using the image coordinates. """
        if self.label is not None:
            self.image.annotate(self.label)

    def __add__(self, other):
        """If given a PosArray then stack on the right, otherwise shift by
        (x, y) coordinates.
        """
        if isinstance(other, (Pos, PosArray)):
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
        if not isinstance(other, (PosArray, Pos)):
            raise TypeError(
                f"Division not supported between PosArray and {type(other)}"
            )
        return self.stack_below(other)

    def __eq__(self, other):
        if not isinstance(other, Pos):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.dx == other.dx
            and self.dy == other.dy
            and self.path == other.path
        )

    def __repr__(self):
        short_path = self.path
        reprstr = f"({self.x:.1f}, {self.y:.1f}) "
        reprstr += f"+({self.dx:.1f}, {self.dy:.1f})"
        reprstr += f" at ::/{short_path}"
        if self.template:
            reprstr += " (template)"
        return reprstr

    @property
    def is_array(self):
        """ Shorthand function that is more extensible than isinstance(self, PosArray). """
        return False


class PosArray(Pos):
    def __init__(self, arr: Tuple[Pos]):
        self.arr = arr if arr is not None else []

    def append(self, other):
        if isinstance(other, Pos):
            self.arr.append(other)
            return self
        elif isinstance(other, PosArray):
            self.arr = self.arr + other.arr
            return self
        if not isinstance(other, (Pos, PosArray)):
            raise TypeError("Unable to append non Pos type")

    def shift_x(self, x_move):
        """ Move all items right by the given amount. """
        [p.shift_x(x_move) for p in self.arr]

    def shift_y(self, y_move):
        """ Move all items down by the given amount. """
        [p.shift_y(y_move) for p in self.arr]

    @property
    def x_min(self):
        """ The smallest x_coordinate in the array. """
        return min(map(lambda p: p.x_min, self.arr))

    @property
    def x_max(self):
        """ The largest x_coordinate in the array. """
        return max(map(lambda p: p.x_max, self.arr))

    @property
    def y_min(self):
        """ The smallest y_coordinate in the array. """
        return min(map(lambda p: p.y_min, self.arr))

    @property
    def y_max(self):
        """ The largest y_coordinate in the array. """
        return max(map(lambda p: p.y_max, self.arr))

    def rescale(self, scale):
        """Rescale all images by a factor.

        The starting points are rescaled as well, about a given point, typically about the
        minimium of the x positions of the included images.
        """

        def rescale_pos_arr(pos_arr):
            for p in pos_arr:
                if p.is_array:
                    # Recursively resize a pos array
                    rescale_pos_arr(p)
                    continue
                from_scale_point_x = p.x_min - self.x_min
                from_scale_point_y = p.y_min - self.y_min

                p.x = self.x_min + from_scale_point_x * scale
                p.y = self.y_min + from_scale_point_y * scale

                p.rescale(scale)

        rescale_pos_arr(self.arr)

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

        rect_kwargs = dict(lw=4, alpha=0.6, ec="black", fc="lightgrey")

        def plot_pos_array(p_array: PosArray, num_offset=0):
            """ Plot all of the leaves within a PosArray. """
            for num, p in enumerate(p_array):
                # If p is a PosArray then pass it again
                if p.is_array:
                    num_offset += plot_pos_array(p, num + num_offset)
                    continue

                #  Else plot the outline of the figure
                ax.add_patch(Rectangle([p.x, p.y], p.dx, p.dy, **rect_kwargs))
                if label:
                    pos = (p.x + p.dx / 2, p.y + p.dy / 2)
                    if not p.path or label == "short":
                        ax.annotate(f"{num+num_offset}", pos, ha="center", va="center")
                    else:
                        ax.annotate(p.path.stem, pos, ha="center", va="center")
            # Ensure that we keep track of the final index
            return num

        plot_pos_array(self)

        pad = 0.1
        ax.set_xlim(self.x_min - pad * x_range, self.x_max + pad * x_range)
        ax.set_ylim(self.y_min - pad * y_range, self.y_max + pad * y_range)
        ax.invert_yaxis()
        ax.set_aspect("equal", "box")
        pt.save(save_path)

    def _normalise_values(self):
        """ Set the x, y, dx, dy positions in the nested array into integers. """

        def normalise_pos_arr(pos_arr: PosArray, attr: str):
            """ Convert the given attr into an integer """
            for p in pos_arr:
                if p.is_array:
                    # Recursively parse ``PosArray``s
                    normalise_pos_arr(p, attr)
                else:
                    # Convert the values on a ``Pos``
                    # The +0.5 rounds the values when they are truncated by int
                    setattr(p, attr, int(getattr(p, attr) + 0.5))
            return

        attrs = ["x", "y", "dx", "dy"]
        [normalise_pos_arr(self.arr, attr) for attr in attrs]

    def populate(self, save_path: Path, final_width: Optional[int] = 1500):
        """ Load the images into the structure. """
        if final_width is not None:
            self.set_width(final_width)
        self._normalise_values()

        # TODO: Currently assuming that the array starts at (0, 0)
        if self.x_min != 0 or self.y_min != 0:
            raise ValueError("Creating image from array with non-zero starting pos")

        im_arr = np.zeros((self.y_range, self.x_range, 4), dtype=np.uint8)
        index = 0

        def populate_in_array(pos_array: PosArray, im: np.ndarray):
            nonlocal index
            for p in pos_array:
                if p.is_array:
                    populate_in_array(p, im)
                    continue

                p.image.resize((p.y_range, p.x_range))
                p.annotate()
                im[p.y_min : p.y_max, p.x_min : p.x_max] = p.image.data
                index = index + 1

        populate_in_array(self.arr, im=im_arr)
        io.imsave(save_path, im_arr, check_contrast=False)

    def __repr__(self):
        return "PosArray: {" + "\n".join([p.__repr__() for p in self]) + "}"

    @property
    def is_array(self):
        """ Shorthand function that is more extensible than isinstance(self, PosArray). """
        return True

    def set_width(self, new_width):
        """ Rescale the image so that final image is the given width"""
        current_width = self.x_range
        scale_factor = new_width / current_width
        self.rescale(scale_factor)


def merge_row(pos_list: List[PosArray]) -> PosArray:
    """ Merge all of the given `PosArray` into a column. """

    def merge_func(x, y):
        return x.stack_right(y)

    return reduce(merge_func, pos_list)


def merge_col(pos_list: List[PosArray]) -> PosArray:
    """ Merge all of the given `PosArray` into a row. """

    def merge_func(x, y):
        return x.stack_below(y)

    return reduce(merge_func, pos_list)
