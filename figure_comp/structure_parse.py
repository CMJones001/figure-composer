#!/usr/bin/env python3

""" Parse a yaml file to create figures.

Config Structure
----------------

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
```

Outline
-------

"""

import yaml
from pathlib import Path
from figure_comp.structure_comp import Row, Col
from icecream import ic
from typing import Optional, Union, List
from dataclasses import dataclass


@dataclass
class ParsedStructure:
    struct: Union[Row, Col]
    leaves: List["ParsedStructure", str]
    options: dict


def parse_file(file_path: Path):
    """ Turn the contents of the given file into a nested Row/Col object. """
    with open(file_path, "r") as f:
        structure_dict = yaml.load(f, Loader=yaml.FullLoader)

    return _parse_section(structure_dict)


def _parse_section(sec):
    """ """
    # Unwrap the list
    if isinstance(sec, list):
        sec = sec[0]

    # Lookup
    structures = dict(Row=Row, Col=Col)

    for key, entry in sec.items():
        if key in ["Row", "Col"]:
            structure = structures[key]
            leaves = _read_leaves(entry)
            options = _get_options(entry)
            return (structure, leaves, options)


def _read_leaves(sec: dict):
    """ Leaves are anything that is not an option. """
    is_leaf = lambda x: not (isinstance(x, dict) and "options" in x)
    return [parse_leaf(row) for row in sec if is_leaf(row)]


def parse_leaf(row: [dict, str]):
    """Read the leaf elements

    Strings are treated as paths, dicts are treated as nested structure
    elements and recursively parsed.
    """
    return row if isinstance(row, str) else _parse_section(row)


def _get_options(sec: dict) -> dict:
    """ Return a dictionary of the options for this layer. """
    for row in sec:
        if isinstance(row, dict) and "options" in row:
            return row["options"]
    return dict()


def assemble_figure(structure):
    """ Turn the yaml given structure into a figure. """


if __name__ == "__main__":
    ic(parse_file())
