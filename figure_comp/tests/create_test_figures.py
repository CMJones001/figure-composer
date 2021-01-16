#!/usr/bin/env python3

""" Create example images for use in tests. """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
import figure_comp.plot_tools as pt

test_dir = Path(__file__).resolve().parent
test_im_dir = test_dir / "test_im"

annotate_kwargs = dict(
    xy=(0.5, 0.5),
    fontsize=40,
    xycoords="figure fraction",
    backgroundcolor="#eee",
    clip_on=True,
)


def create_square_plots(fig_width: float = 4.0):
    """ Create four square plots """
    rc("font", family=r"serif", size=10)
    # Example hist plot
    fig, ax = pt.create_axes(1, fig_width=fig_width)
    count_one = np.random.randn(200)
    count_two = np.random.randn(200) + 0.4
    ax.hist([count_one, count_two], bins=10)
    ax.set_ylabel("Count")
    ax.set_xlabel("Position")
    ax.annotate(1, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "square-im-1.png")

    fig, ax = pt.create_axes(1, fig_width=fig_width)
    x_scale = np.linspace(0, 4 * np.pi, 200)
    y_scale = 2 * np.sin(x_scale)
    y_beat = y_scale * np.sin(x_scale * 0.4 + 0.1)
    ax.plot(x_scale, y_scale, "g--")
    ax.plot(x_scale, y_beat, "k-", lw=1.2)
    ax.set_ylabel("Amplitidue")
    ax.set_xlabel("Time")
    ax.annotate(2, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "square-im-2.png")

    fig, ax = pt.create_axes(1, fig_width=fig_width)
    x_scale = np.linspace(0, 4 * np.pi, 200)
    y_scale = 2 * np.sin(x_scale)
    y_beat = y_scale * np.cos(x_scale * 1.6 + 0.3)
    ax.plot(x_scale, y_scale, "r--")
    ax.plot(x_scale, y_beat, "b-", lw=1.2)
    ax.set_ylabel("Amplitidue")
    ax.set_xlabel("Time")
    ax.annotate(3, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "square-im-3.png")

    fig, ax = pt.create_axes(
        1, fig_width=fig_width, subplot_kw=dict(projection="polar")
    )
    x_scale = np.linspace(0, 2 * np.pi, 200)
    y_scale = 1.0 + np.sin(4 * x_scale) * 0.2
    ax.plot(x_scale, y_scale, "g--")
    ax.annotate(4, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "square-im-4.png", despine=False)

    fig, ax = pt.create_axes(
        1, fig_width=fig_width, subplot_kw=dict(projection="polar")
    )
    x_scale = np.linspace(0, 2 * np.pi, 200)
    y_scale = 1.0 + np.sin(6 * x_scale) * 0.2
    ax.plot(x_scale, y_scale, "m-")
    ax.annotate(5, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "square-im-5.png", despine=False)


def create_rect_plots(fig_width: float = 4.0):
    """ Create four square plots """
    rc("font", family=r"serif", size=10)
    # Example hist plot
    fig, ax = pt.create_axes(1, fig_width=fig_width, aspect=2.0)
    count_one = np.random.randn(200)
    count_two = np.random.randn(200) + 0.4
    ax.hist([count_one, count_two], bins=10)
    ax.set_ylabel("Count")
    ax.set_xlabel("Position")
    ax.annotate(1, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "rect-im-1.png")

    fig, ax = pt.create_axes(1, fig_width=fig_width, aspect=2.0)
    x_scale = np.linspace(0, 4 * np.pi, 200)
    y_scale = 2 * np.sin(x_scale)
    y_beat = y_scale * np.sin(x_scale * 0.4 + 0.1)
    ax.plot(x_scale, y_scale, "g--")
    ax.plot(x_scale, y_beat, "k-", lw=1.2)
    ax.set_ylabel("Amplitidue")
    ax.set_xlabel("Time")
    ax.annotate(2, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "rect-im-2.png")

    fig, ax = pt.create_axes(1, fig_width=fig_width, aspect=2.0)
    x_scale = np.linspace(0, 4 * np.pi, 200)
    y_scale = 2 * np.sin(x_scale)
    y_beat = y_scale * np.cos(x_scale * 1.6 + 0.3)
    ax.plot(x_scale, y_scale, "r--")
    ax.plot(x_scale, y_beat, "b-", lw=1.2)
    ax.set_ylabel("Amplitidue")
    ax.set_xlabel("Time")
    ax.annotate(3, **annotate_kwargs)
    pt.save_figure_and_trim(test_im_dir / "rect-im-3.png")


if __name__ == "__main__":
    create_square_plots()
    create_rect_plots()
