# Figure Composer

This program is designed to automate the layout and labelling of simple figures, in order to save repeated trips to inkscape or powerpoint during figure creation and improvement. 

## Quick start

The `create_figure.py` script needs to be given two arguments:

    ./create_figure.py <config_path> <figure_out_path>
    
the first of these arguments, `<config_path>`, is the path to a configuration describing the structure of the figure. The second argument, `<figure_out_path>` is the path to the created figure. 

## Figure Structure 
The layout of the figure is given as nested series of `Row`s and `Column`s as described in the given configuration file, laid out in a `.yaml` format. Images within a `Row` are scaled to equalise the height while keeping the aspect ratio fixed. 

## Example layouts
### Simple layout
A simple layout of two figures side-by-side is given below:

    - Row:
        - tests/test_im/square-im-1.png
        - tests/test_im/rect-im-1.png

    - Options:
        - format_str: "{index+1}."
        - pos: "0.05, 0.05"
        - size: 20

Also present is a global `Options` block, containing information common to all the sub-figures. 

- `format_str`: string used to create the label in the  sub-figure. `index` is substituted into the figure label, this counts onwards from 0 across the figures. 
- `pos`: A tuple of two floats in `(x, y)` coordinates for the label, relative to subfigure. Values must be in the range `0 < x, y, 1`. 
- `size`: Font size of the labels.

### Overriding options

Options may be over-ridden on a image by image basis: 

    - Row:
        - tests/test_im/square-im-1.png: {text: "A!"}
        - tests/test_im/align.png: {size: 50, pos: "0.50, 0.50"}

    - Options:
        - format_str: "{index+1}."
        - pos: "0.05, 0.05"
        - size: 20

The additional arguments are given as a dictionary, note the `:` following the path to the figure.

### Nested figures

`Row` or `Col` may be nested in the same way as other images.

    - Row:
        - tests/test_im/square-im-1.png: {text: "A!"}
        - Col:
            - tests/test_im/align.png: {size: 50, pos: "0.50, 0.50"}
            - Row:
                - tests/test_im/square-im-3.png
                - tests/test_im/square-im-4.png
    - Options:
        - format_str: "{index+1}."
        - pos: "0.05, 0.05"
        - size: 20

