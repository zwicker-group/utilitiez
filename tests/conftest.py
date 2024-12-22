"""This file is used to configure the test environment when running py.test.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing import (
    set_font_settings_for_testing,
    set_reproducibility_for_testing,
)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Helper function adjusting environment before and after tests."""
    # ensure we use the Agg backend, so figures are not displayed
    plt.switch_backend("agg")
    mpl.rcdefaults()  # Start with all defaults

    set_font_settings_for_testing()
    set_reproducibility_for_testing()

    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")


@pytest.fixture(autouse=False, name="rng")
def init_random_number_generators():
    """Get a random number generator and set the seed of the random number generator.

    The function returns an instance of :func:`~numpy.random.default_rng()` and
    initializes the default generators of both :mod:`numpy` and :mod:`numba`.
    """
    return np.random.default_rng(0)
