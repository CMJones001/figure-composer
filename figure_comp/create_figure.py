#!/usr/bin/env python3

import figure_comp.structure_parse as sp
import argh
from pathlib import Path

""" Manager for the figure composer.

This script takes a path to a configuration file and a save name as an argument.

The configuration file consists of nested "leaves". Each of these leaves may be a path to
an image or nested Row/Column that contains more leaves. The top level is always a Row or
Column.

Examples
========

A simple row with two figures:

```
- Row:
  - /path/one
  - /path/two
  - options:
      max_size: 20
      new_size: 45
```

Two columns with two images each, surrounding a fifth image in a row.

```
- Row:
  - Col:
    - /path/one
    - /path/two
  - /path/five
  - Col:
    - /path/three
    - /path/four
    - options:
        width: 15
  - options:
     max_size: 20
     new_size: 45

"""


def main(
    configuration_path: Path = "broken_config.yaml",
    save_path: Path = "/tmp/debug-figure.png",
    dry: bool = False,
):
    """ Create the figure from the description in the given configuration file. """
    # Convert the yaml into a structure schemematic

    parsed_structure = sp.parse_file(configuration_path)

    # Convert the schemematic into a figure structure
    assembled_figure = parsed_structure.assemble_figure().run()

    if dry:
        assembled_figure.sketch(save_path, label=True)
    else:
        assembled_figure.populate(save_path, final_width=1200)


if __name__ == "__main__":
    argh.dispatch_command(main)
