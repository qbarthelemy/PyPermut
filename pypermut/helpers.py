"""Helpers."""

import sys

import numpy as np


def check_array(X, X_name):
    """Check dimensions of a single sample.

    Parameters
    ----------
    X : array_like
        Sample to check.
    X_name : str
        Name of sample.

    Returns
    -------
    X : array_like, shape (n_meas, n_vars)
        Sample checked.
    """
    X = np.asarray(X)

    if X.ndim > 2:
        raise ValueError(
            "Invalid number of dimensions ndim={} for input {}."
            .format(X.ndim, X_name)
        )

    if X.ndim != 2:
        X = np.atleast_2d(X).T

    return X


def check_paired_arrays(X, Y):
    """Check dimensions of paired samples.

    Parameters
    ----------
    X : array_like
        First paired sample to check.
    Y : array_like
        Second paired sample to check.

    Returns
    -------
    X : array_like, shape (n_meas, n_vars)
        First paired sample checked.
    Y : array_like, shape (n_meas, n_vars)
        Second paired sample checked.
    """
    X = check_array(X, "X")
    Y = check_array(Y, "Y")

    if X.shape != Y.shape:
        raise ValueError(
            "Inputs X and Y do not have compatible dimensions: "
            "X is of dimension {} while Y is {}.".format(X.shape, Y.shape)
        )

    return X, Y


def check_unpaired_arrays(X, Y):
    """Check the second dimension of unpaired samples.

    Parameters
    ----------
    X : array_like
        First unpaired sample to check.
    Y : array_like
        Second unpaired sample to check.

    Returns
    -------
    X : array_like, shape (n_meas_X, n_vars)
        First unpaired sample checked.
    Y : array_like, shape (n_meas_Y, n_vars)
        Second unpaired sample checked.
    """
    X = check_array(X, "X")
    Y = check_array(Y, "Y")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Inputs X and Y do not have the same number of variables: "
            "X is of dimension {} while Y is {}.".format(X.shape, Y.shape)
        )

    return X, Y


def check_groups(X, n_groups_min, check_n_meas=False):
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

    Returns
    -------
    X : array_like
        Sample checked.
    """
    n_groups = len(X)
    if n_groups < n_groups_min:
        raise ValueError(
            f"Need at least {n_groups_min} groups, got {n_groups}."
        )

    #X = list(map(np.asarray, X))
    X = [check_array(x, "") for x in X]

    n_meas, n_vars = X[0].shape
    for i in range(1, n_groups):
        if check_n_meas and X[i].shape[0] != n_meas:
            raise ValueError("Samples have not the same number of measurements.")
        if X[i].shape[1] != n_vars:
            raise ValueError("Samples have not the same number of variables.")

    return X


def get_list_meas(X):
    """Get the number of measurements for each sample.

    Parameters
    ----------
    X : list of array
        The different samples.
        For each sample, the first dimension represents the measurements
        (which can be different for each sample), and the second dimension
        represents different variables.

    Returns
    -------
    list_meas : list
        Number of measurements for each sample.
    """
    list_meas = np.asarray(list(map(len, X)))
    return list_meas


def get_list_ind_meas(list_meas):
    """Get the indices of each group after concatenation.

    Parameters
    ----------
    list_meas : list of int
        Number of measurements for each group.

    Returns
    -------
    list_ind_meas : list of int
        Indices of each group after concatenation.
    """
    list_ind_meas = np.insert(np.cumsum(list_meas), 0, 0)
    return list_ind_meas


def check_n_permutations(n_permutations):
    """Check the parameter n_permutations.

    Parameters
    ----------
    n_permutations : int
        Number of permutations for the permutation test.
    """
    if not isinstance(n_permutations, int):
        raise ValueError("Parameter n_perms_requested must be an integer.")
    if n_permutations < 1:
        raise ValueError("Parameter n_perms_requested must be at least 1.")


def check_permutations(n_perms_requested, n_perms_max, with_replacement):
    """Check the requested permutations.

    Parameters
    ----------
    n_perms_requested : int | "all"
        Number of permutations requested for the test.
    n_perms_max : int
        Maximum number of possible permutations for the test.
    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n_perms_requested is "all".

    Returns
    -------
    perms : array_like
        Permutations for the test.
    n_perms : int
        Number of permutations for the test.
    with_replacement : bool
        Boolean for the chosen bootstrap strategy: with replacement, or without
        replacement.
    """
    if n_perms_requested == "all" or n_perms_requested >= n_perms_max:
        # => exact test, with all permutations
        perms = range(0, n_perms_max)  # from 0, to include null permutation
        n_perms = len(perms)
        with_replacement = False

    else:
        check_n_permutations(n_perms_requested)

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
