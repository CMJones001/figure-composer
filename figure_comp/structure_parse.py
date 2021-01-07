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

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml
from icecream import ic
from skimage import io

from figure_comp.figure_rescale import Image
from figure_comp.structure_comp import Col, Row

Leaf = Union["ParsedStructure", str]


@dataclass
class ParsedStructure:
    struct: Union[Row, Col]
    leaves: List[Leaf]
    options: dict = field(default_factory=dict())

    def as_tuple(self):
        """ Recursively return the structure as tuple. """
        parse_leaf = lambda l: l if isinstance(l, str) else l.as_tuple()
        parsed_leaves = [parse_leaf(l) for l in self.leaves]
        return (self.struct, parsed_leaves, self.options)

    def assemble_figure(self, draft: bool = False):
        """Turn the structure into a figure.

        Parameters
        ----------
        draft: bool
            If True then create placeholder image if an image cannot be found at the path,
            otherwise raise an error.
        """

        def _parse_leaf(leaf: Leaf):
            """If the leaf if a path then resolve then turn this into an image, otherwise
            turn this into a structure."""
            if isinstance(leaf, str):
                image_path = Path(leaf).resolve()
                if not image_path.is_file():
                    if draft:
                        raise NotImplementedError("Finish draft mode")
                    else:
                        raise FileNotFoundError(f"Unable to find image {leaf}")
                return Image(io.imread(image_path), image_path)
            else:
                return leaf.assemble_figure(draft=draft)

        return self.struct([_parse_leaf(leaf) for leaf in self.leaves], **self.options)


def parse_file(file_path: Path):
    """ Turn the contents of the given file into a nested Row/Col object. """
    with open(file_path, "r") as f:
        structure_dict = yaml.load(f, Loader=yaml.FullLoader)

    return _parse_section(structure_dict)


def _parse_section(sec):
    """ """
    # Unwrap the list, required for the top level
    if isinstance(sec, list):
        sec = sec[0]

    # Lookup table between string and objects
    # Might be overkill for now, but scales better
    structures = dict(Row=Row, Col=Col)

    for key, entry in sec.items():
        if key in ["Row", "Col"]:
            structure = structures[key]
            leaves = _read_leaves(entry)
            options = _get_options(entry)
            return ParsedStructure(structure, leaves, options)


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
            if row["options"] is None:
                raise ValueError("Empty option row parsed, check alignment.")
            return row["options"]
    return dict()


if __name__ == "__main__":
    ic(parse_file())
