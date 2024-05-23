""" Core functions for permutations. """

from itertools import combinations
from math import factorial
import numpy as np
from . import helpers


def permute_measurements(Y, x,
                         n_permutations,
                         with_replacement,
                         stat_func,
                         var_func,
                         side):
    """Permute two samples along measurement.

    This generic function permutes two samples along measurement dimension.

    The number of permutations is: n_meas!

    Parameters
    ----------
    Y : array_like, shape (n_meas, n_vars)
        A first sample, with the first dimension representing the measurements
        (measures along time for example), and the second dimension
        representing different variables.

    x : array_like, shape (n_meas,)
        Another sample, same number of measurements as Y.

    n_permutations : int | 'all'
        Number of permutations for the permutation test.
        If n_permutations is 'all', all possible permutations are tested.

    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n_permutations is 'all'.

    stat_func : callable
        Function to compute the multivariate statistics (see module mstats),
        with signature `stat_func(C)`, where C is the horizontal concatenation
        of samples x and Y.

    var_func : callable
        Function to estimate the null distribution across the several
        variables.

    side : {'one', 'two'}
        Side of the test:

        * 'one' for a one-sided test (right side),
        * 'two' for a two-sided test.

    Returns
    -------
    null_dist : array, shape (n_permutations,)
        Vector containing the statistics of the null distribution.
    """
    n_meas = Y.shape[0]

    n_perms_max = count_permutation_measurements(n_meas)
    perms, n_perms, with_replacement = helpers._check_permutations(
        n_perms_requested=n_permutations,
        n_perms_max=n_perms_max,
        with_replacement=with_replacement,
    )

    # loop on permutations to sample the null distribution
    null_dist = np.empty(n_perms, dtype=float)

    for i_perm in range(n_perms):
        if with_replacement:
            permuted_indices = np.random.permutation(n_meas)
        else:
            permuted_indices = get_permutation_measurements(
                n_meas,
                perms[i_perm],
            )
        xperm = x[permuted_indices]  # permute x along measurements

        stat = stat_func(np.c_[xperm, Y])
        if side == 'two':  # for a two-sided test
            stat = np.abs(stat)
        null_dist[i_perm] = var_func(stat)

    return null_dist


def permute_paired_samples(X, Y,
                           n_permutations,
                           with_replacement,
                           stat_func,
                           var_func,
                           side,
                           **kwargs):
# TODO: generalization to S paired samples
    """Permute two paired samples.

    This generic function permutes two paired samples X and Y:
    it flips randomly each pair of multivariate measurements.
    It applies random coeffs -1 or +1 on differences X - Y, along the
    measurement dimension.

    The number of permutations is: 2**n_meas

    Parameters
    ----------
    X : array_like, shape (n_meas, n_vars)
        A first sample, with the first dimension representing the measurements,
        and the second dimension representing different variables.

    Y : array_like, shape (n_meas, n_vars)
        A second sample, same dimensions as X.

    n_permutations : int | 'all'
        Number of permutations for the permutation test.
        If n_permutations is 'all', all possible permutations are tested.

    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n_permutations is 'all'.

    stat_func : callable
        Function to compute the multivariate statistics (see module mstats),
        with signature `stat_func(X-Y)`.

    var_func : callable
        Function to estimate the null distribution across the several
        variables.

    side : {'one', 'two'}
        Side of the test:

        * 'one' for a one-sided test (right side),
        * 'two' for a two-sided test.

    Returns
    -------
    null_dist : array, shape (n_permutations,)
        Vector containing the statistics of the null distribution.
    """
    D = X - Y
    n_meas = D.shape[0]

    n_perms_max = count_permutations_paired_samples(2, n_meas)
    perms, n_perms, with_replacement = helpers._check_permutations(
        n_perms_requested=n_permutations,
        n_perms_max=n_perms_max,
        with_replacement=with_replacement,
    )

    # loop on permutations to sample the null distribution
    null_dist = np.empty(n_perms, dtype=float)

    for i_perm in range(n_perms):
        if with_replacement:
            perm_coeffs = np.random.choice(
                [1, -1],
                size=(n_meas, 1),
                replace=True)
        else:
            perm_coeffs = get_permutation_2_paired_samples(
                n_meas,
                perms[i_perm],
            )
        # permute each pair: apply random coeffs -1 or +1 on differences
        Dperm = np.tile(perm_coeffs, D.shape[1]) * D

        stat = stat_func(Dperm, **kwargs)
        if side == 'two':  # for a two-sided test
            stat = np.abs(stat)
        null_dist[i_perm] = var_func(stat)

    return null_dist


def permute_unpaired_samples(args,
                             n_permutations,
                             with_replacement,
                             stat_func,
                             var_func,
                             side):
    """Permute unpaired samples.

    This generic function permutes S unpaired samples:
    it vertically concatenates samples into C, and then, it permutes C along
    its measurement dimension.

    For S samples X1 ... XS, the number of permutations is:
    (n_meas_X1 + ... + n_meas_XS)! / (n_meas_X1! * ... * n_meas_XS!)

    Parameters
    ----------
    X1, X2, ... : list of array
        List containing the different samples.
        For each sample, the first dimension represents the measurements
        (which can be different for each sample), and the second dimension
        represents different variables (identical for all samples).

    n_permutations : int | 'all'
        Number of permutations for the permutation test.
        If n_permutations is 'all', all possible permutations are tested.

    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement (currently, available only for two samples).
        Unused if n_permutations is 'all'.

    stat_func : callable
        Function to compute the multivariate statistics (see module mstats),
        with signature `stat_func(C, list_meas)`, where C is the vertical
        concatenation of samples, and list_meas is the list of the number of
        measurements for each sample.

    var_func : callable
        Function to estimate the null distribution across the several
        variables.

    side : {'one', 'two'}
        Side of the test:

        * 'one' for a one-sided test (right side),
        * 'two' for a two-sided test.

    Returns
    -------
    null_dist : array, shape (n_permutations,)
        Vector containing the statistics of the null distribution.
    """
    # number of measurements for each sample / group
    list_meas = np.asarray(list(map(len, args)))
    C = np.concatenate(args, axis=0)
    C -= C.mean(axis=0, keepdims=True)  # centering improves num. stability
    n_meas = C.shape[0]

    n_perms_max = count_permutations_unpaired_samples(list_meas)
    perms, n_perms, with_replacement = helpers._check_permutations(
        n_perms_requested=n_permutations,
        n_perms_max=n_perms_max,
        with_replacement=with_replacement,
    )
    if not with_replacement:
        if len(args) != 2:
            raise ValueError('Without replacement strategy is available only '
                             'for two samples.')
        combs = list(combinations(range(n_meas), len(args[0])))

    # loop on permutations to sample the null distribution
    null_dist = np.empty(n_perms, dtype=float)

    for i_perm in range(n_perms):
        if with_replacement:
            permuted_indices = np.random.permutation(n_meas)
        else:
            permuted_indices = get_permutation_unpaired_samples(
                list_meas,
                list(combs[perms[i_perm]]),
            )
        Cperm = C[permuted_indices]  # permute measurements between samples

        stat = stat_func(Cperm, list_meas)
        if side == 'two':  # for a two-sided test
            stat = np.abs(stat)
        null_dist[i_perm] = var_func(stat)

    return null_dist


def count_permutation_measurements(n_meas):
    """Compute the number of unique permutations along measurements.

    This function compute the number of unique permutations along measurements.

    For n_meas measurements, the number of possible permutations is: n_meas!

    Parameters
    ----------
    n_meas : int
        Number of measurements of samples.

    Returns
    -------
    n_perms_max : int
        Number of possible permutations along measurements.
    """
    n_perms_max = factorial(n_meas)

    return n_perms_max


def count_permutations_paired_samples(n_samples, n_meas):
    """Compute the number of unique permutations for paired samples.

    This function compute the number of unique permutations for paired samples.

    For n_samples samples of size n_meas, the number of possible permutations
    is: (n_samples!)**n_meas.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    n_meas : int
        Number of measurements of samples.

    Returns
    -------
    n_perms_max : int
        Number of possible permutations for paired samples.
    """
    n_perms_max = (factorial(n_samples))**n_meas

    return n_perms_max


def count_permutations_unpaired_samples(list_meas):
    """Compute the number of unique permutations for unpaired samples.

    This function compute the number of unique permutations for S unpaired
    samples.

    For S samples X1 ... XS, the number of possible permutations is:
    (n_meas_X1 + ... + n_meas_XS)! / (n_meas_X1! * ... * n_meas_XS!).

    Parameters
    ----------
    list_meas : list of int
        List of number of measurements for each sample.

    Returns
    -------
    n_perms_max : int
        Number of possible permutations for unpaired samples.
    """
    num = factorial(np.sum(list_meas, dtype=int))
    denom = 1
    for n_meas in list_meas:
        denom *= factorial(n_meas)
    n_perms_max = num // denom

    return n_perms_max


def get_permutation_measurements(n_meas, perm_number):
    """Return the list of indices to permute samples along measurements.

    This function returns a list of indices to permute samples along
    measurements, according to a permutation number.

    This function returns a given permutation.
    The null permutation corresponds to perm_number == 0.
    The reverse permutation corresponds to perm_number == n_meas!-1.

    Parameters
    ----------
    n_meas : int
        Number of measurements in samples.

    perm_number : int
        Number indicating the permutation index, from 0 to n_meas!-1.

    Returns
    -------
    permuted_indices : list, length(n_meas)
        The list of permuted indices, for the permutation number.
    """
    sequence = range(n_meas)
    level = len(sequence)
    level_size = factorial(level)
    if perm_number < 0 or perm_number >= level_size:
        raise IndexError('Permutation number {} is out of range [0 ... {}].'
                         .format(perm_number, level_size))

    indices = list(range(level))
    permutation = []

    while level > 0:
        level_size = level_size // level
        current_index = perm_number // level_size
        permutation.append(indices.pop(current_index))
        perm_number = perm_number % level_size
        level -= 1

    return [sequence[i] for i in permutation]


def get_permutation_2_paired_samples(n_meas, perm_number):
    """Return coefficients to permute two paired samples.

    This function returns coefficients to permute two paired samples,
    according to a permutation number.

    This function returns a given permutation.
    The null permutation corresponds to perm_number == 0.
    The full permutation corresponds to perm_number == 2**n_meas-1.

    Parameters
    ----------
    n_meas : int
        Number of measurements of paired samples.

    perm_number : int
        Number indicating the permutation index, from 0 to 2**n_meas-1.

    Returns
    -------
    perm_coeffs : array_like, shape (n_meas, 1)
        Array of permutation coefficients, containing 1 and -1.
    """
    if perm_number < 0 or perm_number >= 2**n_meas:
        raise IndexError('Permutation number {} is out of range [0 ... {}].'
                         .format(perm_number, 2**n_meas))
    # transform number into binary
    bin_coeffs = np.fromiter(
        np.binary_repr(perm_number, width=n_meas),
        dtype=int,
    )
    # transform binaries 0/1 into coefficients 1/-1
    perm_coeffs = 1 - 2 * np.array(bin_coeffs)[:, np.newaxis]

    return perm_coeffs


def get_permutation_unpaired_samples(list_meas, permutated_inds_X):
# TODO: generalization to more than 2 samples; and the signature should be:
# get_permutation_unpaired_samples(list_meas, perm_number)
# requiring to get combinations by their indices.
    """Return the indices to permute two unpaired samples.

    This function returns the indices to permute two unpaired samples X and Y,
    according to permutation indices of X.

    Parameters
    ----------
    list_meas : list of two int
        List of number of measurements for samples X and Y.

    permutated_inds_X : list, length(n_meas_X)
        List of permutation indices for X.

    Returns
    -------
    permutated_inds : list, length(n_meas_X + n_meas_Y)
        List of permutation indices to apply to the concatenation of X and Y.
    """
    assert len(list_meas) == 2, 'Function valid only for 2 samples.'
    permutated_inds_Y = [
        ind for ind in range(list_meas[0] + list_meas[1])
        if ind not in permutated_inds_X
    ]
    permutated_inds = permutated_inds_X + permutated_inds_Y

    return permutated_inds

