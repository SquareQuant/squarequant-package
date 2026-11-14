"""Small shared validators for statistical inputs and observation counts."""

from decimal import Decimal
from numbers import Integral, Real

import numpy as np


def positive_integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def probability(value, name: str = "confidence") -> float:
    result = finite_real(value, name)
    if not 0 < result < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return result
