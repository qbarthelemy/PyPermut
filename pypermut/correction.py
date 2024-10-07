"""Correction functions."""

import numpy as np


def correct_bonferroni(p_vals, n_tests=None):
    """Correct p-values by Bonferroni's method.

    Correction for multiple tests, using Bonferroni's method [1]_:
    multiply p-values by the number of tests.

    Parameters
    ----------
    p_vals : array, shape (n_vals,)
        The p-values.

    n_tests : None | int, default=None
        Number of tests. If None, n_tests is set to n_vals.

    Returns
    -------
    p_vals_corrected : array, shape (n_vals,)
        The p-values corrected by Bonferroni's method.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bonferroni_correction
    """
    if n_tests is None:
        n_tests = len(p_vals)
    return p_vals * n_tests


def correct_holm(p_vals):
    """Correct p-values by Holm's step-down method.

    Correction for multiple tests, using Holm's step-down method [1]_.

    Parameters
    ----------
    p_vals : array, shape (n_vals,)
        The p-values.

    Returns
    -------
    p_vals_corrected : array, shape (n_vals,)
        The p-values corrected by Holm's method.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    """
    n_tests = len(p_vals)
    sortind = np.argsort(p_vals)
    p_vals = np.take(p_vals, sortind)

    pvals_corrected_raw = p_vals * np.arange(n_tests, 0, -1)
    pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[sortind] = pvals_corrected

    return pvals_corrected_
