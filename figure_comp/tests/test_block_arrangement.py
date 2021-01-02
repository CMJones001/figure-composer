#!/usr/bin/env python3

import unittest as u

import figure_comp.block_arrangement as ba
import logging
from icecream import ic

# logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")


class Test_CreateBoxes(u.TestCase):
    def test_square_creation(self):
        box = ba._Box(1, 1)
        aspect_expected = 1
        aspect_test = box.aspect

        self.assertEqual(aspect_test, aspect_expected)


class Test_FillFigures(u.TestCase):
    def test_simple_fill(self):
        """ Simplest case of equally sized boxes. """
        box_list = [ba._Box(2, 2, f"{i}") for i in range(5)]
        figure_width = 6

        fig = ba.FillFigures(box_list, figure_width)

        n_rows_expected = 2
        n_cols_expected = 3

        n_cols_test = fig.n_cols
        n_rows_test = fig.n_rows

        self.assertEqual(n_cols_test, n_cols_expected)
        self.assertEqual(n_rows_test, n_rows_expected)

        ic(fig.boxes)
        for row in fig.rows:
            for box in row:
                print(box)
            print()


if __name__ == "__main__":
    u.main()
