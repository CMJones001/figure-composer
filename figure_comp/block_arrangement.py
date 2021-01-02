#!/usr/bin/env python

""" Outline for arangement of figures. """

from dataclasses import dataclass
from typing import Optional, List
import logging


@dataclass
class _Box:
    """Template for the wrapper class.

    This should contain some method for loading the information, but also storing
    positional information about its own position in the grid.
    """

    width: int
    height: int
    label: Optional[str] = None

    def __post_init__(self):
        self.aspect = self.width / self.height
        self.pos = None
        self.index = None

    def __str__(self):
        return f"Box size {self.width},{self.height} at {self.index}; {self.pos}"

    def __repr__(self):
        return f"Box({self.width},{self.height} at {self.index}; {self.pos})"


class _ArangeFigures:
    def __init__(self, boxes: List[_Box]):
        self.boxes = boxes
        self.max_fig_width = max(map(lambda x: x.width, self.boxes))


class FillFigures(_ArangeFigures):
    def __init__(self, Boxes: List[_Box], figure_width: float):
        """ Simply fit the figures in the available width. """
        super().__init__(Boxes)
        self.figure_width = figure_width

        if self.figure_width < self.max_fig_width:
            raise ValueError(
                "box width is greater than figure width"
                f"{box.width} > {self.figure_width}"
            )

        self.fill_boxes()

    def fill_boxes(self):
        """Simply fit the figures in the available width.

        This works simply enough, however, we also want to include the starting positions
        of each item in the box.

        Another class?
        """
        self.rows = []
        n_row = 0
        current_width = 0
        current_row = []

        for n_box, box in enumerate(self.boxes):
            new_width = current_width + box.width
            if new_width <= self.figure_width:
                logging.debug(
                    f"figure {n_box} at ({current_width} -> {new_width}) on {n_row}"
                )

                box.pos = (current_width, new_width)
                box.index = (len(current_row), n_row)

                current_width = new_width
                current_row.append(box)

            else:
                n_row += 1
                current_width = box.width
                box.pos = (0, current_width)
                box.index = (0, n_row)
                logging.debug(
                    f"Creating new row, box {n_box} at (0 -> {current_width})"
                )
                self.rows.append(current_row)
                current_row = [box]

        self.rows.append(current_row)

    @property
    def n_rows(self):
        return len(self.rows)

    @property
    def n_cols(self):
        return max(map(len, self.rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    box_list = [_Box(2, 2, f"{i}") for i in range(5)]
    figure_width = 6

    FillFigures(box_list, figure_width)
