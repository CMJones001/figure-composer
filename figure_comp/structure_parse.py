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

import numpy as np
import yaml
from icecream import ic

from figure_comp.coordinate_tracking import Pos
from figure_comp.load_image import Label
from figure_comp.structure_comp import Col, Row, _Container


def _read_branch(branch, dry=False, **kwargs):
    struct = []
    header = "Row" if "Row" in branch else "Col"
    header_struct = Row if header == "Row" else Col

    # Provide defaults for general keywords
    # Overkill for now, but can be expanded later
    defaults = dict(default_labels=None)
    opts = {**defaults, **kwargs}

    for entry in branch[header]:
        if isinstance(entry, str):
            struct.append(_parse_path(entry, **opts))
        elif _is_subbranch(entry):
            struct.append(_read_branch(entry, dry=dry, **opts))
        elif isinstance(entry, dict):
            struct.append(_parse_complex_path(entry, **opts))
        else:
            raise ValueError("Unable to parse branch")

    if dry:
        return (header, struct)
    return header_struct(struct)


def parse_file(file_path: Path, default_labels=None):
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

    return parse_yaml(structure_dict, default_labels=default_labels)


def parse_yaml(structure_dict: dict, dry=False, default_labels=None) -> _Container:
    """ Convert the yaml dict into a Row or Col container. """
    # Remove top level list if passed
    if isinstance(structure_dict, list):
        structure_dict = structure_dict[0]
    return _read_branch(structure_dict, dry=dry, default_labels=default_labels)


def _parse_complex_path(leaf, default_labels=None):
    """ Parse a path with label overrides. """

    # NB. We don't iterate here, there is only one entry, but this splits the
    # path (key) and option dict (values) nicely.
    for path, label_override in leaf.items():

        # Pos has to be treated before passing into label
        if "pos" in label_override:
            pos_str = label_override["pos"].strip("()")
            pos = np.fromstring(pos_str, sep=", ")
            label_override["pos"] = pos

        label_func = next(default_labels)
        label = label_func(**label_override)
        return Pos(path, label)


def _parse_path(leaf, default_labels=None):
    """ Parse a simple path into a path. """
    if default_labels is not None:
        label = next(default_labels)()
        ic(label)
    else:
        label = None
    return Pos(leaf, label)


def _is_subbranch(leaf):
    return "Col" in leaf or "Row" in leaf


if __name__ == "__main__":
    ic(parse_file())
