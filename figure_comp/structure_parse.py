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

from pathlib import Path

import yaml
from icecream import ic

from figure_comp.coordinate_tracking import Pos
from figure_comp.load_image import Label
from figure_comp.structure_comp import Col, Row, _Container
import numpy as np


def _read_branch(branch, dry=False):
    struct = []
    header = "Row" if "Row" in branch else "Col"
    header_struct = Row if header == "Row" else Col

    for entry in branch[header]:
        if isinstance(entry, str):
            struct.append(_parse_path(entry))
        elif _is_subbranch(entry):
            struct.append(_read_branch(entry, dry=dry))
        elif isinstance(entry, dict):
            struct.append(_parse_complex_path(entry))
        else:
            raise ValueError("Unable to parse branch")

    if dry:
        return (header, struct)
    return header_struct(struct)


def parse_file(file_path: Path):
    """ Turn the contents of the given file into a nested Row/Col object. """
    try:
        with open(file_path, "r") as f:
            structure_dict = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.parser.ParserError:
        print(f"Unable to parse configuration file: {file_path}")
        raise SystemExit(1)
    except yaml.scanner.ScannerError:
        print(f"Malformed configuration file: {file_path}")
        print("Check for colons after first option line and indents without dashes! ")
        raise SystemExit(1)

    return parse_yaml(structure_dict)


def parse_yaml(structure_dict: dict, dry=False) -> _Container:
    """ Convert the yaml dict into a Row or Col container. """
    # Remove top level list if passed
    if isinstance(structure_dict, list):
        structure_dict = structure_dict[0]
    return _read_branch(structure_dict, dry=dry)


def _parse_complex_path(leaf):
    """ Parse a path with label_opt: """
    for path, label_opt in leaf.items():
        if "pos" in label_opt:
            pos_str = label_opt["pos"].strip("()")
            pos = np.fromstring(pos_str, sep=", ")
            label_opt.pop("pos")
        else:
            pos = (0.1, 0.1)
        label = Label(pos=pos, **label_opt)
        return Pos(path, label)


def _parse_path(leaf):
    """ Parse a simple path into a path. """
    return Pos(leaf)


def _is_subbranch(leaf):
    return "Col" in leaf or "Row" in leaf


if __name__ == "__main__":
    ic(parse_file())
