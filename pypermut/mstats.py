"""Functions to compute statistics of multivariate samples."""

import numpy as np
import scipy.stats as stats

from . import helpers


def pearsonr(xY):
    """Pearson correlation coefficient.

    This function computes the Pearson correlation coefficient between a
    multivariate sample Y and a univariate sample x.

    Parameters
    ----------
    xY : array, shape (n_meas, 1 + n_vars)
        The horizontal concatenation of y and Y.

    Returns
    -------
    r : array, shape (n_vars,)
        The statistics r between x and variables of Y.
    """
    return np.corrcoef(xY, rowvar=False)[0, 1:]


def spearmanr(xY):
    """Spearman correlation coefficient.

    This function computes the Spearman correlation coefficient between a
    multivariate sample Y and a univariate sample x.

    Parameters
    ----------
    xY : array, shape (n_meas, 1 + n_vars)
        The horizontal concatenation of y and Y.

    Returns
    -------
    rho : array, shape (n_vars,)
        The statistics rho between x and variables of Y.
    """
    rho = stats.spearmanr(xY).correlation
    if xY.shape[1] == 2:
        # different behaviour when there is only two variables
        return np.array([rho], dtype=float)
    else:
        # extract the first row without the diagonal element
        return rho[0, 1:]


def spearmanr_fast(xY):
    """Spearman correlation coefficient, faster version.

    This function computes the Spearman correlation coefficient between a
    multivariate sample Y and a univariate sample x.
    Faster version, without checking on data (we have to ensure the data is on
    the correct shape).

    Parameters
    ----------
    xY : array, shape (n_meas, 1 + n_vars)
        The horizontal concatenation of y and Y.

    Returns
    -------
    rho : array, shape (n_vars,)
        The statistics rho between x and variables of Y.
    """
    ranked = np.apply_along_axis(stats.rankdata, 0, xY)
    rho = np.corrcoef(ranked, None, rowvar=False)
    return rho[0, 1:]


def spearmanr_fast_unique(xY):
    """Spearman correlation coefficient, faster version for unique data.

    This function computes the Spearman correlation coefficient between a
    multivariate sample Y and a univariate sample x.
    Faster version, that needs the data to be unique.

    Parameters
    ----------
    xY : array, shape (n_meas, 1 + n_vars)
        The horizontal concatenation of y and Y.

    Returns
    -------
    rho : array, shape (n_vars,)
        The statistics rho between x and variables of Y.
    """
    ranked = np.apply_along_axis(stats.rankdata, 0, xY)
    D = ranked - ranked[:, 0, np.newaxis]
    n_meas = D.shape[0]
    rho = 1 - 6*np.sum(D * D, axis=0) / (n_meas * (n_meas*n_meas - 1))
    return rho[1:]


def studentt_rel(D):
    """Student t statistic for related samples.

    This function computes the Student t statistic for related multivariate
    samples X and Y.

    Parameters
    ----------
    D : array, shape (n_meas, n_vars)
        The difference between sets of measurements: D = X - Y.

    Returns
    -------
    t : array, shape (n_vars,)
        The statistics t between variables of X and Y.
    """
    D_mean = np.mean(D, axis=0)
    D_var = np.var(D, axis=0, ddof=1)
    denom = np.sqrt(D_var / D.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(D_mean, denom)
    return t


def studentt_ind(C, list_meas):
    """Student t statistic for independent samples.

    This function computes the Student t statistic for independent multivariate
    samples X and Y.

    Parameters
    ----------
    C : array, shape (n_meas_X + n_meas_Y, n_vars)
        The vertical concatenation of sets of measurements X and Y.

    list_meas : list of int
        Number of measurements for each sample: [n_meas_X, n_meas_Y].

    Returns
    -------
    t : array, shape (n_vars,)
        The statistics t between variables of X and Y.
    """
    n_meas_X, n_meas_Y = list_meas[0], list_meas[1]
    D = np.mean(C[0:n_meas_X], axis=0) - np.mean(C[n_meas_X:], axis=0)
    var1 = np.var(C[0:n_meas_X], axis=0, ddof=1)
    var2 = np.var(C[n_meas_X:], axis=0, ddof=1)
    svar = ((n_meas_X-1) * var1 + (n_meas_Y-1) * var2) \
           / (n_meas_X + n_meas_Y - 2.0)
    denom = np.sqrt(svar * (1.0/n_meas_X + 1.0/n_meas_Y))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(D, denom)
    return t


def welcht_ind(C, list_meas):
    """Welch t statistic for independent samples.

    This function computes the Welch t statistic for independent multivariate
    samples X and Y. It does not assume equal variance.

    Parameters
    ----------
    C : array, shape (n_meas_X + n_meas_Y, n_vars)
        The vertical concatenation of sets of measurements X and Y.

    list_meas : list of int
        Number of measurements for each sample: [n_meas_X, n_meas_Y].

    Returns
    -------
    t : array, shape (n_vars,)
        The statistics t between variables of X and Y.
    """
    n_meas_X, n_meas_Y = list_meas[0], list_meas[1]
    D = np.mean(C[0:n_meas_X], axis=0) - np.mean(C[n_meas_X:], axis=0)
    var1 = np.var(C[0:n_meas_X], axis=0, ddof=1)
    var2 = np.var(C[n_meas_X:], axis=0, ddof=1)
    denom = np.sqrt(var1/n_meas_X + var2/n_meas_Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(D, denom)
    return t


def wilcoxon(D, zero_method):
    """Wilcoxon T statistic for paired samples.

    This function computes the Wilcoxon T statistic for paired multivariate
    samples X and Y.

    Parameters
    ----------
    D : array, shape (n_meas, n_vars)
        The difference between sets of measurements: D = X - Y.

    zero_method : {"pratt", "wilcox", "zsplit"}
        Method for zero-differences processing.

    Returns
    -------
    T : array, shape (n_vars,)
        The statistics T (not W) between variables of X and Y.
    """
    if zero_method == "wilcox":
        # WARNING: non-multivariate processing for this option
        T = np.zeros(D.shape[1])
        for v, d in enumerate(D.T): # loop on variables...
            d = np.compress(np.not_equal(d, 0), d, axis=-1)

            r = stats.rankdata(np.abs(d))
            r_plus = np.sum((d > 0) * r)
            r_minus = np.sum((d < 0) * r)

            T[v] = min(r_plus, r_minus)
        return T

    r = np.apply_along_axis(stats.rankdata, 0, np.abs(D))
    r_plus = np.sum((D > 0) * r, axis=0)
    r_minus = np.sum((D < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((D == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = np.minimum(r_plus, r_minus)
    return T


def mannwhitneyu(C, list_meas):
    """Mann-Whitney U statistic for unpaired samples.

    This function computes the Mann-Whitney U statistic for unpaired
    multivariate samples X and Y.

    Warning: statistic U is not computed like scipy.stats.mannwhitneyu.

    Parameters
    ----------
    C : array, shape (n_meas_X + n_meas_Y, n_vars)
        The vertical concatenation of sets of measurements X and Y.

    list_meas : list of int
        Number of measurements for each sample: [n_meas_X, n_meas_Y].

    Returns
    -------
    U : array, shape (n_vars,)
        The statistics U between variables of X and Y.
    """
    n_meas_X, n_meas_Y = list_meas[0], list_meas[1]
    ranked = np.apply_along_axis(stats.rankdata, 0, C)
    ranks_X = ranked[0:n_meas_X]
    U_X = np.sum(ranks_X, axis=0) - (n_meas_X * (n_meas_X + 1)) / 2.
    U_Y = n_meas_X * n_meas_Y - U_X  # because U_X + U_Y = n_meas_X * n_meas_Y
    U = np.minimum(U_X, U_Y)
    return U


def f_oneway(C, list_meas):
    """Fisher F statistic for one-way ANOVA for independent samples.

    This function computes the Fisher F statistic for one-way ANOVA applied on
    independent samples, with several variables.

    Parameters
    ----------
    C : array, shape (n_meas, n_vars)
        The vertical concatenation of groups of measurements, already centered
        for each variable.

    list_meas : list of int
        Number of measurements for each group.

    Returns
    -------
    F : array, shape (n_vars,)
        The statistics F between variables of groups.
    """
    # indices of each group after vertical concatenation
    list_inds = helpers.get_list_ind_meas(list_meas)
    n_groups = len(list_meas)
    n_meas = list_inds[-1]  # total number of measurements

    normalized_ss = np.sum(C, 0)**2 / n_meas
    sstot = np.sum(C**2, 0) - normalized_ss

    ssbn = 0
    for i in range(n_groups):
        ss = np.sum(C[list_inds[i]:list_inds[i+1]], 0)**2
        ssbn += ss / list_meas[i]
    ssbn -= normalized_ss
    msb = ssbn / (n_groups - 1)
    msw = (sstot - ssbn) / (n_meas - n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = msb / msw
    return np.atleast_1d(F)


def kruskal(C, list_meas):
    """Kruskal–Wallis H statistic for independent samples.

    This function computes the Kruskal–Wallis H statistic for independent
    multivariate samples.

    Parameters
    ----------
    C : array, shape (n_meas, n_vars)
        The vertical concatenation of groups of measurements.

    list_meas : list of int
        Number of measurements for each group.

    Returns
    -------
    H : array, shape (n_vars,)
        The statistics H between variables of groups.
    """
    # indices of each group after vertical concatenation
    list_inds = helpers.get_list_ind_meas(list_meas)
    n_groups = len(list_meas)
    n_meas = list_inds[-1]  # total number of measurements

    ranked = np.apply_along_axis(stats.rankdata, 0, C)
    ties = np.apply_along_axis(stats.tiecorrect, 0, ranked)
    if np.any(ties == 0):
        raise ValueError("All numbers are identical in Kruskal-Wallis test.")

    ssbn = 0
    for i in range(n_groups):
        ss = np.sum(ranked[list_inds[i]:list_inds[i+1]], 0)**2
        ssbn += ss / list_meas[i]
    H = 12.0 / (n_meas * (n_meas+1)) * ssbn - 3 * (n_meas+1)
    H /= ties
    return np.atleast_1d(H)


def friedmanchisquare(Data):
    """Friedman chi-squared statistic for paired samples.

    This function computes the Friedman chi-squared statistic for paired
    multivariate samples.

    Parameters
    ----------
    Data : array, shape(n_meas, n_groups, n_vars)
        The concatenation of groups of measurements.

    Returns
    -------
    chisq : array, shape (n_vars,)
        The statistics chi-square between variables of groups.
    """
    n_meas, n_groups, n_vars = Data.shape

    ranked = np.apply_along_axis(stats.rankdata, 1, Data)
    reps = np.apply_along_axis(stats.find_repeats, 1, ranked)
    repnums = reps[:, 1, ...]

    ties = np.zeros(n_vars)
    for i in range(n_meas):
        for t in repnums[i]:
            ties += t * (t*t - 1)
    c = 1 - ties / (n_groups * (n_groups*n_groups - 1) * n_meas)

    ssbn = np.sum(ranked.sum(axis=0)**2, axis=0)
    chisq = (12.0 / (n_groups*n_meas*(n_groups+1)) * ssbn
             - 3*n_meas*(n_groups+1)) / c
    return chisq
