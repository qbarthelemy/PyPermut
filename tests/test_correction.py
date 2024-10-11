"""Tests for module correction.

To execute tests:
>>> pytest -k test_correction
"""

import numpy as np
import pytest

import pypermut.correction as correction


###############################################################################


@pytest.mark.parametrize("n_vals", [5, 10, 15])
def test_correct_bonferroni(n_vals):
    """Test correct_bonferroni."""
    p_vals = np.random.uniform(low=0.0, high=1.0, size=n_vals)
    p_vals_corrected = correction.correct_bonferroni(p_vals)
    assert p_vals.shape == p_vals_corrected.shape


@pytest.mark.parametrize("n_vals", [5, 10, 15])
def test_correct_holm(n_vals):
    """Test correct_holm."""
    p_vals = np.random.uniform(low=0.0, high=1.0, size=n_vals)
    p_vals_corrected = correction.correct_holm(p_vals)
    assert p_vals.shape == p_vals_corrected.shape
