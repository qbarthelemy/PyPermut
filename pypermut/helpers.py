"""Helpers."""

import sys

import numpy as np


def _check_array(X, X_name):
    """This function checks dimensions of a single sample."""
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
    """This function checks dimensions of paired samples."""
    X = _check_array(X, "X")
    Y = _check_array(Y, "Y")

    if X.shape != Y.shape:
        raise ValueError(
            "Inputs X and Y do not have compatible dimensions: "
            "X is of dimension {} while Y is {}.".format(X.shape, Y.shape)
        )

    return X, Y


def _check_unpaired_arrays(X, Y):
    """This function checks the second dimension of unpaired samples."""
    X = _check_array(X, "X")
    Y = _check_array(Y, "Y")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Inputs X and Y do not have the same number of variables: "
            "X is of dimension {} while Y is {}.".format(X.shape, Y.shape)
        )

    return X, Y


def _check_n_permutations(n_permutations):
    """This function checks the parameter n_permutations."""
    if not isinstance(n_permutations, int):
        raise ValueError("Parameter n_perms_requested must be an integer.")
    if n_permutations < 1:
        raise ValueError("Parameter n_perms_requested must be at least 1.")


def _check_permutations(n_perms_requested, n_perms_max, with_replacement):
    """This function checks the requested permutations."""

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
