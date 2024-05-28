"""Helpers."""

import sys

import numpy as np


def _check_array(X, X_name):
    """Check dimensions of a single sample."""
    X = np.asarray(X)

    if X.ndim > 2:
        raise ValueError(
            "Invalid number of dimensions ndim={} for input {}."
            .format(X.ndim, X_name)
        )

    if X.ndim != 2:
        X = np.atleast_2d(X).T

    return X


def _check_paired_arrays(X, Y):
    """Check dimensions of paired samples."""
    X = _check_array(X, "X")
    Y = _check_array(Y, "Y")

    if X.shape != Y.shape:
        raise ValueError(
            "Inputs X and Y do not have compatible dimensions: "
            "X is of dimension {} while Y is {}.".format(X.shape, Y.shape)
        )

    return X, Y


def _check_unpaired_arrays(X, Y):
    """Check the second dimension of unpaired samples."""
    X = _check_array(X, "X")
    Y = _check_array(Y, "Y")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Inputs X and Y do not have the same number of variables: "
            "X is of dimension {} while Y is {}.".format(X.shape, Y.shape)
        )

    return X, Y


def _check_groups(X, n_groups_min, check_n_meas=False):
    """Check groups.

    Parameters
    ----------
    X : array_like
        The samples for each group.
        For each sample, the first dimension represents the measurements, and
        the second dimension represents different variables.

    n_groups_min : int
        Minimum number of groups.

    check_n_meas : bool, default=False
        Choose if samples have the same number of measurements.
    """
    n_groups = len(X)
    if n_groups < n_groups_min:
        raise ValueError(
            f"Need at least {n_groups_min} groups, got {n_groups}."
        )

    #X = list(map(np.asarray, X))
    X = [_check_array(x, "") for x in X]

    n_meas, n_vars = X[0].shape
    for i in range(1, n_groups):
        if check_n_meas and X[i].shape[0] != n_meas:
            raise ValueError("Samples have not the same number of measurements.")
        if X[i].shape[1] != n_vars:
            raise ValueError("Samples have not the same number of variables.")

    return X


def _get_list_meas(X):
    """Get the number of measurements for each sample / group."""
    list_meas = np.asarray(list(map(len, X)))
    return list_meas


def _get_list_ind_meas(list_meas):
    """Get the indices of each group after concatenation."""
    list_ind_meas = np.insert(np.cumsum(list_meas), 0, 0)
    return list_ind_meas


def _check_n_permutations(n_permutations):
    """Check the parameter n_permutations."""
    if not isinstance(n_permutations, int):
        raise ValueError("Parameter n_perms_requested must be an integer.")
    if n_permutations < 1:
        raise ValueError("Parameter n_perms_requested must be at least 1.")


def _check_permutations(n_perms_requested, n_perms_max, with_replacement):
    """Check the requested permutations."""

    if n_perms_requested == "all" or n_perms_requested >= n_perms_max:
        # => exact test, with all permutations
        perms = range(0, n_perms_max)  # from 0, to include null permutation
        n_perms = len(perms)
        with_replacement = False

    else:
        _check_n_permutations(n_perms_requested)

        n_perms = n_perms_requested
        if not with_replacement and n_perms_max < sys.maxsize:
            # => permutation test, using bootstrap without replacement
            # we uniformly sample a subset of all possible permutations
            perms = np.random.choice(
                np.arange(0, n_perms_max),
                size=n_perms,
                replace=False,
            )
        else:
            # => permutation test, using bootstrap with replacement
            perms = []
            with_replacement = True

    return perms, n_perms, with_replacement
