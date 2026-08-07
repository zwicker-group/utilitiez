#!/usr/bin/env python3 -u

import sys
import timeit
from pathlib import Path

import numpy as np

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))

TEMPLATES = {
    "numpy": """
from numpy.random import uniform
from utilitiez import xlogx
def fun(x):
    return xlogx(x, diff={diff}, threshold={threshold})
x = uniform(0, 1, size={size})
fun(x)  # call once for caching
""",
    "numba": """
from numpy.random import uniform
from numba import jit
from utilitiez import xlogx
@jit
def fun(x):
    return xlogx(x, diff={diff}, threshold={threshold})
x = uniform(0, 1, size={size})
fun(x)  # call once for caching
""",
    "jax": """
from numpy.random import uniform
from jax import jit
from utilitiez import xlogx
@jit
def fun(x):
    return xlogx(x, diff={diff}, threshold={threshold})
x = uniform(0, 1, size={size})
fun(x)  # call once for caching
""",
}


def run_performance_test(
    backend: str,
    num: int = 1000,
    size: int = 1000,
    *,
    diff: int = 0,
    threshold: float = 0.0,
):
    """Run the performance test"""
    print(f"{backend}:")

    # test specific class
    sys.stdout.write(f"  {backend:5s} (specific): ")
    time = timeit.repeat(
        "fun(x)",
        setup=TEMPLATES[backend].format(size=size, diff=diff, threshold=threshold),
        number=num,
    )
    print(f"{int(num / np.mean(time)):>8d}")


def main():
    """Main routine testing the performance."""
    print("Reports the number of calls per second. Larger is better.\n")

    for backend in ["numpy", "numba", "jax"]:
        run_performance_test(backend, diff=1, threshold=0.1)


if __name__ == "__main__":
    main()
