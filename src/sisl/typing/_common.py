# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Union

import scipy.sparse as sps

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from sisl import Geometry, Lattice

__all__ = [
    "Coord",
    "CoordOrScalar",
    "FuncType",
    "KPoint",
    "LatticeOrGeometry",
    "SeqFloat",
    "SeqOrScalarFloat",
    "SparseMatrix",
    "SparseMatrixExt",
]


SeqFloat = Sequence[float]
SeqOrScalarFloat = Union[float, SeqFloat]

Coord = SeqFloat
CoordOrScalar = Union[float, Coord]

KPoint = Sequence[float]

# Short for *any* function
FuncType = Callable[..., Any]

LatticeOrGeometry = Union[
    "Lattice",
    "Geometry",
]

if hasattr(sps, "sparray"):
    SparseMatrixExt = Union[
        sps.spmatrix,
        sps.sparray,
    ]
else:
    SparseMatrixExt = Union[sps.spmatrix,]

SparseMatrix = Union[
    SparseMatrixExt,
    "SparseCSR",
]
