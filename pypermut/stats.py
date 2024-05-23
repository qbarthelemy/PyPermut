"""Permutations for statistical tests.

Random state can be fixed before calling permutation tests,
using np.random.seed().
"""

import warnings

import numpy as np
from scipy.stats import percentileofscore

from . import helpers
from . import core
from . import mstats


def permutation_corr(Y, *,
                     x=None,
                     n=10000,
                     with_replacement=True,
                     corr="pearson",
                     side="one",
                     return_dist=False):
# TODO: allow x to be multivariate -> X
    """Rmax permutation test.

    This function performs a Rmax permutation test from correlations between a
    multivariate sample Y and a univariate sample x.

    Parameters
    ----------
    Y : array_like, shape (n_meas, n_vars)
        A first sample, with the first dimension representing the measurements
        (measures along time for example), and the second dimension
        representing different variables.

    x : None | array_like, shape (n_meas,), default=None
        Another sample, same number of measurements as Y. By default, it is a
        monotonous vector giving a longitudinal test. When x is a time vector,
        the null hypothesis is that there is no time effect in measures.

    n : int | "all", default=10000
        Number of permutations for the permutation test.
        If n is "all", all possible permutations are tested, giving exact test.

    with_replacement : bool, default=True
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n is "all".

    corr : {"pearson", "spearman"}, default "pearson"
        Define the correlation type:

        * "pearson" for the Pearson product-moment correlation coefficient r,
        * "spearman" for the Spearman rank-order correlation coefficient rho.

    side : {"one", "two"}, default="one"
        Side of the test:

        * "one" for a one-sided test (right side),
        * "two" for a two-sided test.

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    Rstats : list of float, length (n_vars)
        The correlation coefficients between variables of X and y.

    pvals : list of float, length (n_vars)
        The p-values computed from Rmax distribution of permuted measurements.

    Rmax : array, shape (n_perms,)
        The Rmax distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    .. [2] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    .. [4] https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient
    """
    Y = helpers._check_array(Y, "Y")
    n_meas = Y.shape[0]

    if x is None:
        x = np.arange(n_meas)  # monotonous vector
    else:
        x = np.asarray(x)
        if x.ndim != 1 or x.shape[0] != n_meas:
            raise ValueError(
                "Inputs Y and x do not have compatible dimensions: "
                "Y is of dimension {} while x is {}.".format(Y.shape, x.shape))

    if corr in ["pearson", "pearsonr"]:
        corr_func = mstats.pearsonr
    elif corr in ["spearman", "spearmanr"]:
        corr_func = mstats.spearmanr
    elif corr == "spearman_fast":
        # check if use _spearman_fast_unique, only when all measures are unique
        y_unique = np.apply_along_axis(lambda col: len(set(col)), 0, Y)
        x_unique = len(set(x))
        if np.all(y_unique == x.shape[0]) and x_unique == x.shape[0]:
            corr_func = mstats.spearmanr_fast_unique
        else:
            warnings.warn("Not all observations are unique, using the "
                "spearman_fast function that is slight slower "
                "(but still fast!)", UserWarning, stacklevel=2)
            corr_func = mstats.spearmanr_fast
    elif corr == "spearman_fast_unique":
        # Warning, only use if you are sure (this is kept for unit tests)
        warnings.warn("Use spearman_fast instead of spearman_fast_unique",
            UserWarning, stacklevel=2)
        corr_func = mstats.spearmanr_fast_unique
    elif corr == "spearman_fast_nonunique":
        # Warning, only use if you are sure (this is kept for unit tests)
        warnings.warn("Use spearman_fast instead of spearman_fast_nonunique",
            UserWarning, stacklevel=2)
        corr_func = mstats.spearmanr_fast
    else:
        raise ValueError(f"Invalid value for corr={corr}.")

    if side not in ["one", "two"]:
        raise ValueError(f"Invalid value for side={side}.")

    # under the null hypothesis, sample the Rmax distribution
    Rmax = core.permute_measurements(
        Y, x,
        n,
        with_replacement,
        stat_func=corr_func,
        var_func=np.max,
        side=side,
    )

    # compute the real R statistics
    Rstats = Rstats_ = corr_func(np.c_[x, Y])
    if side == "two":
        Rstats_ = np.abs(Rstats_)
    # compare them to the Rmax distribution with a right-sided test,
    # because significance is obtained for high R stats
    pvals = np.array([
        (100 - percentileofscore(Rmax, R, kind="strict")) / 100
        for R in Rstats_
    ])

    if return_dist:
        return Rstats, pvals, Rmax
    else:
        return Rstats, pvals


def permutation_pearsonr(*args, **kwargs):
    raise ValueError("This function does not exist. "
                     "Use permutation_corr with corr='pearson'.")


def permutation_spearmanr(*args, **kwargs):
    raise ValueError("This function does not exist. "
                     "Use permutation_corr with corr='spearman'.")


def permutation_ttest_rel(X, Y, *,
                          n=10000,
                          with_replacement=True,
                          side="one",
                          return_dist=False):
    """tmax permutation test for related / paired samples.

    This function performs a tmax permutation test from Student's t-tests,
    applied on related / paired samples with several variables.

    Parameters
    ----------
    X : array_like, shape (n_meas, n_vars)
        A first sample, with the first dimension representing the measurements,
        and the second dimension representing different variables.

    Y : array_like, shape (n_meas, n_vars)
        A second sample, same dimensions as X.

    n : int | "all", default=10000
        Number of permutations for the permutation test.
        If n is "all", all possible permutations are tested, giving exact test.

    with_replacement : bool, default=True
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n is "all".

    side : {"one", "two"}, default="one"
        Side of the test:

        * "one" for a one-sided test (right side),
        * "two" for a two-sided test.

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    tstats : list of int, length (n_vars)
        List of statistics t between variables of X and Y.

    pvals : list of float, length (n_vars)
        List of p-values computed from tmax distribution of permuted
        measurements, for right-sided tests.

    tmax : array, shape (n_perms,)
        The tmax distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    .. [2] https://en.wikipedia.org/wiki/Student's_t-test#Dependent_t-test_for_paired_samples
    """
    X, Y = helpers._check_paired_arrays(X, Y)

    if side not in ["one", "two"]:
        raise ValueError(f"Invalid value for side={side}.")

    # under the null hypothesis, sample the tmax distribution
    tmax = core.permute_paired_samples(
        X, Y,
        n_permutations=n,
        with_replacement=with_replacement,
        stat_func=mstats.studentt_rel,
        var_func=np.max,
        side=side,
    )

    # compute the real t statistics
    tstats = tstats_ = mstats.studentt_rel(X - Y)
    if side == "two":
        tstats_ = np.abs(tstats_)
    # compare them to the tmax distribution with a right-sided test,
    # because significance is obtained for high t stats
    pvals = np.array([
        (100 - percentileofscore(tmax, t, kind="strict")) / 100
        for t in tstats_
    ])

    if return_dist:
        return tstats, pvals, tmax
    else:
        return tstats, pvals


def permutation_ttest_ind(X, Y, *,
                          n=10000,
                          with_replacement=True,
                          side="one",
                          equal_var=True,
                          return_dist=False):
    """tmax permutation test for independent / unpaired samples.

    This function performs a tmax permutation test from Student's t-tests,
    applied on independent / unpaired samples with several variables.

    Parameters
    ----------
    X : array_like, shape (n_meas_X, n_vars)
        A first sample, with the first dimension representing the measurements,
        and the second dimension representing different variables.

    Y : array_like, shape (n_meas_Y, n_vars)
        A second sample, same number of variables as X.

    n : int | "all", default=10000
        Number of permutations for the permutation test.
        If n is "all", all possible permutations are tested, giving exact test.

    with_replacement : bool, default=True
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n is "all".

    side : {"one", "two"}, default="one"
        Side of the test:

        * "one" for a one-sided test (right side),
        * "two" for a two-sided test.

    equal_var : bool, default=True
        If True, it performs the standard independent two samples test that
        assumes equal variances.
        If False, it performs the Welch's t-test, which does not assume equal
        variance.

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    tstats : list of int, length (n_vars)
        List of statistics t between variables of X and Y.

    pvals : list of float, length (n_vars)
        List of p-values computed from tmax distribution of permuted
        measurements, for right-sided tests.

    tmax : array, shape (n_perms,)
        The tmax distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    .. [2] https://en.wikipedia.org/wiki/Student's_t-test#Independent_two-sample_t-test
    .. [3] https://en.wikipedia.org/wiki/Welch's_t-test
    """
    X, Y = helpers._check_unpaired_arrays(X, Y)

    if side not in ["one", "two"]:
        raise ValueError(f"Invalid value for side={side}.")

    if equal_var:
        stat_func = mstats.studentt_ind
    else:
        stat_func = mstats.welcht_ind

    # under the null hypothesis, sample the tmax distribution
    tmax = core.permute_unpaired_samples(
        [X, Y],
        n_permutations=n,
        with_replacement=with_replacement,
        stat_func=stat_func,
        var_func=np.max,
        side=side,
    )

    # compute the real t statistics
    tstats = tstats_ = stat_func(
        np.concatenate((X, Y), axis=0),
        [X.shape[0], Y.shape[0]],
    )
    if side == "two":
        tstats_ = np.abs(tstats_)
    # compare it to the tmax distribution with a right-sided test,
    # because significance is obtained for high t stats
    pvals = np.array([
        (100 - percentileofscore(tmax, t, kind="strict")) / 100
        for t in tstats_
    ])

    if return_dist:
        return tstats, pvals, tmax
    else:
        return tstats, pvals


def permutation_wilcoxon(X, Y, *,
                         n=10000,
                         with_replacement=True,
                         zero_method="wilcox",
                         return_dist=False):
    """Tmin permutation test.

    This function performs a Tmin permutation test from Wilcoxon T tests
    (also called Wilcoxon signed-rank test), applied on paired samples with
    several variables.

    It is a Tmin test, because low values of T are required for significance.

    Parameters
    ----------
    X : array_like, shape (n_meas, n_vars)
        A first sample, with the first dimension representing the measurements,
        and the second dimension representing different variables.

    Y : array_like, shape (n_meas, n_vars)
        A second sample, same dimensions as X.

    n : int | "all", default=10000
        Number of permutations for the permutation test.
        If n is "all", all possible permutations are tested, giving exact test.

    with_replacement : bool, default=True
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n is "all".

    zero_method : {"pratt", "wilcox", "zsplit"}, default="wilcox"
        Method for zero-differences processing.

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    Tstat : list of int, length (n_vars)
        List of statistics T between variables of X and Y.

    pvals : list of float, length (n_vars)
        List of p-values computed from Tmin distribution of permuted
        measurements, for left-sided tests.

    Tmin : array, shape (n_perms,)
        The Tmin distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    .. [2] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """
    X, Y = helpers._check_paired_arrays(X, Y)

    if not zero_method in ["wilcox", "pratt", "zsplit"]:
        raise ValueError(f"Invalid value for zero_method={zero_method}.")

    # under the null hypothesis, sample the Tmin distribution
    Tmin = core.permute_paired_samples(
        X, Y,
        n,
        with_replacement,
        stat_func=mstats.wilcoxon,
        var_func=np.min,
        side="one",
        zero_method=zero_method,
    )

    # compute the real T statistics
    Tstats = mstats.wilcoxon(X - Y, zero_method)
    # compare it to the Tmin distribution with a left-sided test,
    # because significance is obtained for low T stats
    pvals = np.array([
        percentileofscore(Tmin, T, kind="weak") / 100 for T in Tstats
    ])

    if return_dist:
        return Tstats, pvals, Tmin
    else:
        return Tstats, pvals


def permutation_mannwhitneyu(X, Y, *,
                             n=10000,
                             with_replacement=True,
                             return_dist=False):
    """Umin permutation test.

    This function performs a Umin permutation test from Mann-Whitney U tests
    (sometimes called Wilcoxon rank-sum tests), applied on unpaired samples
    with several variables.

    It is a Umin test, because low values of U are required for significance.

    Parameters
    ----------
    X : array_like, shape (n_meas_X, n_vars)
        A first sample, with the first dimension representing the measurements,
        and the second dimension representing different variables.

    Y : array_like, shape (n_meas_Y, n_vars)
        A second sample, same number of variables as X.

    n : int | "all", default=10000
        Number of permutations for the permutation test.
        If n is "all", all possible permutations are tested, giving exact test.

    with_replacement : bool, default=True
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement. Unused if n is "all".

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    Ustat : list of int, length (n_vars)
        List of statistics U between variables of X and Y.
        Warning: these statistics U are not computed like scipy.stats.mannwhitneyu.

    pvals : list of float, length (n_vars)
        List of p-values computed from Umin distribution of permuted
        measurements, for left-sided tests.

    Umin : array, shape (n_perms,)
        The Umin distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    .. [2] https://en.wikipedia.org/wiki/Mann-Whitney_U_test
    """
    X, Y = helpers._check_unpaired_arrays(X, Y)

    # under the null hypothesis, sample the Umin distribution
    Umin = core.permute_unpaired_samples(
        [X, Y],
        n_permutations=n,
        with_replacement=with_replacement,
        stat_func=mstats.mannwhitneyu,
        var_func=np.min,
        side="one",
    )

    # compute the real U statistics
    Ustats = mstats.mannwhitneyu(
        np.concatenate((X, Y), axis=0),
        [X.shape[0], Y.shape[0]],
    )
    # compare it to the Umin distribution with a left-sided test,
    # because significance is obtained for low U stats
    pvals = np.array([
        percentileofscore(Umin, U, kind="weak") / 100 for U in Ustats
    ])

    if return_dist:
        return Ustats, pvals, Umin
    else:
        return Ustats, pvals


def permutation_f_oneway(*args, n=10000, return_dist=False):
# TODO: allow n="all"
    """Fmax permutation test.

    This function performs a Fmax permutation test from one-way ANOVAs,
    applied on independent / unpaired samples with several variables.

    Parameters
    ----------
    X1, X2, ... : array_like
        The samples for each group, at least 2.
        For each sample, the first dimension represents the measurements
        (which can be different for each sample), and the second dimension
        represents different variables (identical for all samples).

    n : int, default=10000
        Number of permutations for the permutation test.

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    Fstats : list of float, length (n_vars)
        The F statistics between variables of samples.

    pvals : list of float, length (n_vars)
        The p-values computed from Fmax distribution of permuted measurements.

    Fmax : array, shape (n_perms,)
        The Fmax distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    .. [2] https://en.wikipedia.org/wiki/Analysis_of_variance
    """
    args = list(map(np.asarray, args))
    if len(args) < 2:
        raise ValueError("Need at least two groups.")

    if not isinstance(n, int):
        raise ValueError("Parameter n must be an integer.")

    # under the null hypothesis, sample the Fmax distribution
    Fmax = core.permute_unpaired_samples(
        args,
        n_permutations=n,
        with_replacement=True,
        stat_func=mstats.f_oneway,
        var_func=np.max,
        side="one",
    )

    # number of measurements for each sample / group
    list_meas = np.asarray(list(map(len, args)))
    # compute the real F statistics
    Fstats = mstats.f_oneway(np.concatenate(args, axis=0), list_meas)
    # compare it to the Fmax distribution with a right-sided test,
    # because significance is obtained for high F stats
    pvals = np.array([
        (100 - percentileofscore(Fmax, F, kind="strict")) / 100 for F in Fstats
    ])

    if return_dist:
        return Fstats, pvals, Fmax
    else:
        return Fstats, pvals


def permutation_kruskal(*args, n=10000, return_dist=False):
# TODO: allow n="all"
    """Hmax permutation test.

    This function performs a Hmax permutation test from Kruskal-Wallis H
    tests (sometimes called one-way ANOVA on ranks), applied on independent
    samples with several variables.

    Parameters
    ----------
    X1, X2, ... : array_like
        The samples for each group, at least 2.
        For each sample, the first dimension represents the measurements
        (which can be different for each sample), and the second dimension
        represents different variables (identical for all samples).

    n : int, default=10000
        Number of permutations for the permutation test.

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    Hstats : list of float, length (n_vars)
        The H statistics between variables of samples.

    pvals : list of float, length (n_vars)
        The p-values computed from Hmax distribution of permuted measurements.

    Hmax : array, shape (n_perms,)
        The Hmax distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    .. [2] https://en.wikipedia.org/wiki/Kruskal-Wallis_one-way_analysis_of_variance
    """
    args = list(map(np.asarray, args))
    if len(args) < 2:
        raise ValueError("Need at least two groups.")

    if not isinstance(n, int):
        raise ValueError("Parameter n must be an integer.")

    # under the null hypothesis, sample the Hmax distribution
    Hmax = core.permute_unpaired_samples(
        args,
        n_permutations=n,
        with_replacement=True,
        stat_func=mstats.kruskal,
        var_func=np.max,
        side="one",
    )

    list_meas = np.asarray(list(map(len, args)))
    # compute the real H statistics
    Hstats = mstats.kruskal(np.concatenate(args, axis=0), list_meas)
    # compare it to the Hmax distribution with a right-sided test,
    # because significance is obtained for high H stats
    pvals = np.array([
        (100 - percentileofscore(Hmax, H, kind="strict")) / 100 for H in Hstats
    ])

    if return_dist:
        return Hstats, pvals, Hmax
    else:
        return Hstats, pvals


def permutation_friedmanchisquare(*args, n=10000, return_dist=False):
# TODO: allow n="all"; and use core.permute_paired_samples after its
# generalization to S samples
    """chi2max permutation test.

    This function performs a chi2max permutation test from Friedman chi2
    (chi-square) tests, applied on paired samples with several variables.

    Parameters
    ----------
    X1, X2, X3, ... : array_like
        The samples for each group, at least 3.
        For each sample, the first dimension represents the measurements
        (identical for all samples), and the second dimension represents
        different variables (identical for all samples).

    n : int, default=10000
        Number of permutations for the permutation test.

    return_dist : bool, default=False
        Boolean to choose if null distribution is added to outputs.

    Returns
    -------
    chi2stats : list of float, length (n_vars)
        The chi2 statistics between variables of samples.

    pvals : list of float, length (n_vars)
        The p-values computed from chi2max distribution of permuted
        measurements.

    chi2max : array, shape (n_perms,)
        The chi2max distribution sampled under null hypothesis.
        Returned only when return_dist is True.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html
    .. [2] https://en.wikipedia.org/wiki/Friedman_test
    """
    args = list(map(np.asarray, args))
    n_groups = len(args)
    if len(args) < 3:
        raise ValueError(f"Need at least three groups, got {n_groups}.")
    if args[0].ndim < 2:
        raise ValueError("Inputs must be at least 2D.")
    n_meas, n_vars = args[0].shape
    for i in range(1, n_groups):
        if args[i].shape != (n_meas, n_vars):
            raise ValueError("Unequal sample shapes in friedmanchisquare.")

    if not isinstance(n, int):
        raise ValueError("Parameter n must be an integer.")

    Data_ = np.dstack(args)  # shape = (n_meas, n_vars, n_groups)
    Data = np.transpose(Data_.astype(float),
                        axes=(0, 2, 1))  # shape = (n_meas, n_groups, n_vars)

    # under the null hypothesis, sample the chi2max distribution
    chi2max = np.empty(n, dtype=float)

    for i_perm in range(n):
        Dataperm = Data.copy()
        for m in range(n_meas):  # for each measure, permute along groups
            permuted_indices = np.random.permutation(n_groups)
            Dataperm[m] = Dataperm[m][permuted_indices]

        stat = mstats.friedmanchisquare(Dataperm)
        chi2max[i_perm] = np.max(stat)

    # compute the real chi2 statistics
    chi2stats = mstats.friedmanchisquare(Data)
    # compare it to the chimax distribution with a right-sided test,
    # because significance is obtained for high chi2 stats
    pvals = np.array([
        (100 - percentileofscore(chi2max, chi2, kind="strict")) / 100
        for chi2 in chi2stats
    ])

    if return_dist:
        return chi2stats, pvals, chi2max
    else:
        return chi2stats, pvals
