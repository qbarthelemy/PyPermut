"""Tests for module stats.

To execute tests:
>>> pytest -k test_stats
"""

import warnings

import numpy as np
import pytest
import scipy.stats as stats

import pypermut.core as core
import pypermut.stats as ppstats

np.random.seed(17)


###############################################################################


@pytest.mark.parametrize("n", range(2, 8))
@pytest.mark.parametrize(
    "corr_func",
    ("pearson", "spearman", "spearman_fast", "spearman_fast_nonunique")
)
def test_permutation_measurements_best(n, corr_func):
    """Test the statistic r and the exact p-value p of correlation tests,
    when no permutation can improve the statistic.

    Parameters
    ----------
    n : int
        Length of samples that will be tested. Do not use values over 8, since
        it would take too long to permute completely.
    corr_func : string
        Define the correlation type:

        * "pearson" for the Pearson product-moment correlation coefficient r,
        * "spearman" for the Spearman rank-order correlation coefficient rho.
    """
    assert n > 1, "Need at least 2 values."

    # silent warning in spearman_fast_non_unique
    if corr_func == "spearman_fast_nonunique":
        warnings.simplefilter("ignore", UserWarning)

    # create a vector that increases monotonically: 0,1,2,...,n-3,n-2,n-1.
    # consequently, no permutation can improve the correlation
    y = np.arange(n)

    # there are n! different permutations
    max_perms = core.count_permutation_measurements(n)

    # use a correlation permutation test with all permutations:
    # it should give a p-value of exactly 1/(n!)
    r, p, dist = ppstats.permutation_corr(
        y,
        n="all",
        side="one",
        corr=corr_func,
        return_dist=True,
    )

    assert dist.shape[0] == max_perms, (
        "Null distribution should contain n! statistics."
    )
    assert np.isclose(r[0], 1., atol=1e-9), (
        "Best permutation case should give r of exactly 1."
    )
    assert np.isclose(p[0], 1./max_perms, atol=1e-9), (
        "Best permutation case should give p-value of exactly 1/(n!)."
    )


@pytest.mark.parametrize("n", range(2, 14))
def test_permutation_paired_samples_best(n):
    """Test the exact p-value of tests for two related samples, when no
    permutation can improve the statistic.

    Parameter
    ---------
    n : int
        Length of samples that will be tested.
    """
    assert n > 1, "Need at least 2 values."

    # create a first random vector x, with positive values
    x = np.random.randn(n) + 5
    x[x < 0] = 0

    # create a second vector y, a quasi-shift of x, but without overlap with x:
    # consequently, no permutation can improve the statistic
    y = x + 0.001 * np.random.randn(n) - 10

    # there are 2^n different permutations
    max_perms = core.count_permutations_paired_samples(2, n)

    # use a Student's permutation t-test for related samples with all possible
    # permutations: it should give a p-value of exactly 1/(2^n)
    t, p, dist = ppstats.permutation_ttest_rel(
        x, y,
        n="all",
        side="one",
        return_dist=True,
    )
    assert dist.shape[0] == max_perms, (
        "Student t-test rel: null distribution should contain 2^n statistics."
    )
    assert np.isclose(p[0], 1./max_perms, atol=1e-9), (
        "Student t-test rel: best permutation case should give p-value of "
        "exactly 1/(2^n)."
    )

    # use a Wilcoxon permutation T test for paired samples with all possible
    # permutations: it should give a p-value of exactly 2/(2^n)
    # (because T stat is computed on the absolute differences,
    # so full permutation gives the same T as null permutation)
    T, p, dist = ppstats.permutation_wilcoxon(
        x, y,
        n="all",
        zero_method="wilcox",
        return_dist=True,
    )
    assert dist.shape[0] == max_perms, (
        "Wilcoxon T test: null distribution should contain 2^n statistics."
    )
    assert np.isclose(p[0], 2./max_perms, atol=1e-9), (
        "Wilcoxon T test: best permutation case should give p-value of exactly"
        " 1/(2^n)."
    )


@pytest.mark.parametrize("n1", range(2, 9))
@pytest.mark.parametrize("n2", range(2, 9))
def test_permutation_unpaired_samples_best(n1, n2):
    """Test the exact p-value of tests for two independent samples, when no
    permutation can improve the statistic.

    Parameters
    ----------
    n1 : int
        Length of first sample that will be tested.
    n2 : int
        Length of second sample that will be tested.
    """
    assert n1 > 1, "Need at least 2 values."
    assert n2 > 1, "Need at least 2 values."

    # create a first random vector x, with positive values
    x = np.random.randn(n1) + 5
    x[x < 0] = 0

    # create a second vector y, with negative values, without overlap with x.
    # consequently, no permutation can improve the statistic
    y = np.random.randn(n2) - 5
    y[y > 0] = 0

    # there are (n1+n2)!/(n1!n2!) different permutations
    max_perms = core.count_permutations_unpaired_samples([n1, n2])

    # use a Student's permutation t-test for independent samples with all
    # possible permutations: it should give a p-value of exactly
    # 1/((n1+n2)!/(n1!n2!))
    t, p, dist = ppstats.permutation_ttest_ind(
        x, y,
        n="all",
        side="one",
        return_dist=True,
    )
    assert dist.shape[0] == max_perms, (
        "Student t-test ind: null distribution should contain "
        "(n1+n2)!/(n1!n2!) statistics."
    )
    # TODO: assert p-value
    # NOT DONE, because percentileofscore randomly detects the presence of stat
    # of null permutation in vector dist...
    # assert np.isclose(p[0], 1./max_perms, atol=1e-9), (
    #     "Student t-test ind: best permutation case should give p-value of "
    #     "exactly 1/((n1+n2)!/(n1!n2!))."
    # )

    # use a Mann-Whitney permutation U test for unpaired samples with all
    # possible permutations: it should give a p-value of exactly
    # 2/((n1+n2)!/(n1!n2!))
    # (because computing U = min(U_x, U_y) gives two identical U)
    U, p, dist = ppstats.permutation_mannwhitneyu(
        x, y,
        n="all",
        return_dist=True,
    )
    assert dist.shape[0] == max_perms, (
        "Mann-Whitney U test: null distribution should contain "
        "(n1+n2)!/(n1!n2!) statistics."
    )
    assert np.isclose(p[0], 2./max_perms, atol=1e-9), (
       "Mann-Whitney U test: best permutation case should give p-value of "
       "exactly 1/((n1+n2)!/(n1!n2!))."
    )


###############################################################################


@pytest.mark.parametrize(
    "corr_func",
    ("pearson", "spearman", "spearman_fast", "spearman_fast_nonunique")
)
@pytest.mark.parametrize("with_replacement", (True, False))
@pytest.mark.parametrize("side", ("one", "two"))
@pytest.mark.parametrize("n_meas", range(2, 5))
@pytest.mark.parametrize("n_vars", range(1, 5))
def test_permutation_corr_args(
    corr_func,
    with_replacement,
    side,
    n_meas,
    n_vars,
):
    """Check permutation_corr arguments.

    Parameters
    ----------
    corr_func : string
        Define the correlation type:

        * "pearson" for the Pearson product-moment correlation coefficient r,
        * "spearman" for the Spearman rank-order correlation coefficient rho.
    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement.
    side : {"one", "two"}
        Side of the test:

        * "one" for a one-sided test (right side),
        * "two" for a two-sided test.
    n_meas : int
        Number of measurements in samples X and Y.
    n_vars : int
        Number of variables in samples X and Y.
    """
    # silent warning in spearman_fast_non_unique
    if corr_func == "spearman_fast_nonunique":
        warnings.simplefilter("ignore", UserWarning)

    if n_vars == 1:
        ppstats.permutation_corr(
            np.random.randn(n_meas),
            x=np.random.randn(n_meas),
            n=1,
            with_replacement=with_replacement,
            corr=corr_func,
            side=side,
        )

    stats, pvals = ppstats.permutation_corr(
        np.random.randn(n_meas, n_vars),
        x=np.random.randn(n_meas),
        n=1,
        with_replacement=with_replacement,
        corr=corr_func,
        side=side,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_corr_errors(n_meas=5, n_vars=3):
    """Check permutation_corr errors."""
    with pytest.raises(ValueError):  # not same n_meas
        ppstats.permutation_corr(
            np.random.randn(n_meas, n_vars),
            x=np.random.randn(n_meas + 1),
            n=1,
        )

    with pytest.raises(ValueError):  # x not univariate
        ppstats.permutation_corr(
            np.random.randn(n_meas, n_vars),
            x=np.random.randn(n_meas, n_vars),
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_corr(
            np.random.randn(n_meas, n_vars, 3),
            n=1,
        )


@pytest.mark.parametrize(
    "corr_func",
    ("pearson", "spearman", "spearman_fast", "spearman_fast_nonunique")
)
def test_permutation_corr_rates(
    corr_func,
    n_reps=20,
    n_meas=25,
    n_vars=10,
    trend_amplitude=3.,
    side="one",
    alpha=0.05,
):
    """Validate the permutation_corr function by generating random samples on
    which statistics and p-values are computed, and checking false positive and
    true positive rates.

    Parameters
    ----------
    corr_func : string
        Define the correlation type:

        * "pearson" for the Pearson product-moment correlation coefficient r,
        * "spearman" for the Spearman rank-order correlation coefficient rho.
    n_reps : int, default=20
        Number of repetitions.
    n_meas : int, default=25
        Number of measurements in samples X and Y.
    n_vars : int, default=10
        Number of variables in samples X and Y.
    trend_amplitude : float, default=3
        Amplitude of the generated temporal trend over unitary flat noise.
    side : string, default="one"
        TODO
    alpha : float, default=0.05
        Type-I error used for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate.
    """
    # silent warning in spearman_fast_non_unique
    if corr_func == "spearman_fast_nonunique":
        warnings.simplefilter("ignore", UserWarning)

    false_positives = []
    true_positives = []

    for i in range(n_reps):
        # generate a random data matrix
        Y = np.random.random((n_meas, n_vars))
        # add tendency only to the last variable:
        # this trend is negative, so should be captured by one-sided test
        Y[:, -1] -= trend_amplitude * np.arange(n_meas, dtype=float) / n_meas
        Y = -Y

        R_pp, pvals = ppstats.permutation_corr(
            Y,
            n=100,
            with_replacement=False,
            side=side,
            corr=corr_func,
        )

        # assert r values with respect SciPy
        x = np.arange(n_meas)
        R_sp = np.zeros(n_vars)
        for v, y in enumerate(Y.T):
            if "pearson" in corr_func:
                r, _ = stats.pearsonr(x, y)
                R_sp[v] = r
            else:
                R_sp[v] = stats.spearmanr(x, y).correlation
        assert np.allclose(R_pp, R_sp, atol=1e-9), (
            "Statistics R should be equivalent to SciPy."
        )

        # the trend was only added to the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha


###############################################################################


@pytest.mark.parametrize("with_replacement", (True, False))
@pytest.mark.parametrize("side", ("one", "two"))
@pytest.mark.parametrize("n_meas", range(1, 5))
@pytest.mark.parametrize("n_vars", range(1, 5))
def test_permutation_ttest_rel_args(
    with_replacement,
    side,
    n_meas,
    n_vars,
):
    """Check permutation_ttest_rel arguments.

    Parameters
    ----------
    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement.
    side : {"one", "two"}
        Side of the test:

        * "one" for a one-sided test (right side),
        * "two" for a two-sided test.
    n_meas : int
        Number of measurements in samples X and Y.
    n_vars : int
        Number of variables in samples X and Y.
    """
    if n_vars == 1:
        ppstats.permutation_ttest_rel(
            np.random.randn(n_meas),
            np.random.randn(n_meas),
            n=1,
            with_replacement=with_replacement,
            side=side,
        )

    stats, pvals = ppstats.permutation_ttest_rel(
        np.random.randn(n_meas, n_vars),
        np.random.randn(n_meas, n_vars),
        n=1,
        with_replacement=with_replacement,
        side=side,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_ttest_rel_errors(n_meas=5, n_vars=3):
    """Check permutation_ttest_rel errors."""
    with pytest.raises(ValueError):  # not same n_meas
        ppstats.permutation_ttest_rel(
            np.random.randn(n_meas, n_vars),
            np.random.randn(n_meas + 1, n_vars),
            n=1,
        )

    with pytest.raises(ValueError):  # not same n_vars
        ppstats.permutation_ttest_rel(
            np.random.randn(n_meas, n_vars),
            np.random.randn(n_meas, n_vars + 1),
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_ttest_rel(
            np.random.randn(n_meas, n_vars, 3),
            np.random.randn(n_meas, n_vars, 3),
            n=1,
        )


def test_permutation_ttest_rel_rates(
    n_reps=50,
    n_meas=100,
    n_vars=10,
    diff_mean=-2.,
    diff_std=0.8,
    alpha=0.05,
):
    """Validate the permutation_ttest_rel function by generating random samples
    on which statistics and p-values are computed, and checking false positive
    and true positive rates.

    H0: difference between pairs follows a Gaussian distribution.

    Parameters
    ----------
    n_reps : int, default=50
        Number of repetitions.
    n_meas : int, default=100
        Number of measurements in samples X and Y.
    n_vars : int, default=10
        Number of variables in samples X and Y.
    diff_mean : float, default=-2.0
        Value of the mean of the distribution in Y different from X.
    diff_std : float, default=0.8
        Value of the standard deviation of the distribution in Y different from
        X.
    alpha : float, default=0.05
        Type-I error use for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate.
    """
    false_positives = []
    true_positives = []

    for i in range(n_reps):
        # X = random Gaussian data, mean=0, std=1
        X = np.random.randn(n_meas, n_vars)
        # Y = X + Gaussian noise, to be compliant with H0
        # ie, pairs difference follows a Gaussian distribution around zero
        # because should give a low value t
        Y = X + 0.1 * np.random.randn(n_meas, n_vars)
        # add an offset to last variable, to reject H0
        # because should give a high value t
        Y[:, -1] = diff_std * np.random.randn(n_meas) + diff_mean

        t_pp, pvals = ppstats.permutation_ttest_rel(
            X, Y,
            n=100,
            with_replacement=False,
            side="one",
        )

        # assert t values with respect SciPy
        t_sp = np.zeros(n_vars)
        for v, (x, y) in enumerate(zip(X.T, Y.T)):
            t_sp[v] = stats.ttest_rel(x, y).statistic
        assert np.allclose(t_pp, t_sp, atol=1e-9), (
            "Statistics t should be equivalent to SciPy."
        )

        # the distribution is different only for the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha


###############################################################################


@pytest.mark.parametrize("with_replacement", (True, False))
@pytest.mark.parametrize("side", ("one", "two"))
@pytest.mark.parametrize("equal_var", (True, False))
@pytest.mark.parametrize("n_meas_X", [5, 6, 7])
@pytest.mark.parametrize("n_meas_Y", [10, 12, 15])
@pytest.mark.parametrize("n_vars", range(1, 5))
def test_permutation_ttest_ind_args(
    with_replacement,
    side,
    equal_var,
    n_meas_X,
    n_meas_Y,
    n_vars,
):
    """Check permutation_ttest_ind arguments.

    Parameters
    ----------
    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement.
    side : {"one", "two"}
        Side of the test:

        * "one" for a one-sided test (right side),
        * "two" for a two-sided test.
    equal_var : bool
        If True, it performs the standard independent two samples test that
        assumes equal variances.
        If False, it performs the Welch's t-test, which does not assume equal
        variance.
    n_meas_X : int
        Number of measurements in X.
    n_meas_Y : int
        Number of measurements in Y.
    n_vars : int
        Number of variables in X and Y.
    """
    if n_vars == 1:
        ppstats.permutation_ttest_ind(
            np.random.randn(n_meas_X),
            np.random.randn(n_meas_Y),
            n=1,
            with_replacement=with_replacement,
            side=side,
            equal_var=equal_var,
        )

    stats, pvals = ppstats.permutation_ttest_ind(
        np.random.randn(n_meas_X, n_vars),
        np.random.randn(n_meas_Y, n_vars),
        n=1,
        with_replacement=with_replacement,
        side=side,
        equal_var=equal_var,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_ttest_ind_errors(n_meas_X=5, n_meas_Y=6, n_vars=3):
    """Check permutation_ttest_ind errors."""
    with pytest.raises(ValueError):  # not same n_vars
        ppstats.permutation_ttest_ind(
            np.random.randn(n_meas_X, n_vars),
            np.random.randn(n_meas_Y, n_vars + 1),
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_ttest_ind(
            np.random.randn(n_meas_X, n_vars, 3),
            np.random.randn(n_meas_Y, n_vars, 3),
            n=1,
        )


@pytest.mark.parametrize("equal_var", (True, False))
def test_permutation_ttest_ind_rates(
    equal_var,
    n_reps=50,
    n_meas_X=100,
    n_meas_Y=110,
    n_vars=10,
    diff_mean=-2.,
    alpha=0.05,
):
    """Validate the permutation_ttest_ind function by generating random samples
    on which statistics and p-values are computed, and checking false positive
    and true positive rates.

    H0: samples are drawn from the same Gaussian distribution, ie with equal
    means and variances (excepted for Welch's version).

    Parameters
    ----------
    equal_var : bool
        If True, it performs the standard independent two samples test that
        assumes equal variances.
        If False, it performs the Welch's t-test, which does not assume equal
        variance.
    n_reps : int, default=50
        Number of repetitions.
    n_meas_X : int, default=100
        Number of measurements in X.
    n_meas_Y : int, default=110
        Number of measurements in Y.
    n_vars : int, default=10
        Number of variables in X and Y.
    diff_mean : float, default=-2.0
        Value of the mean of the distribution in Y different from X.
    alpha : float, default=0.05
        Type-I error use for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate
    """
    false_positives = []
    true_positives = []

    if equal_var:
        std = 1
    else:
        # std = random uniform value between 1 and 10
        std = 10 * np.random.rand(n_vars) + 1

    for i in range(n_reps):
        # X = random Gaussian data, mean=0, std=1
        X = np.random.randn(n_meas_X, n_vars)
        # Y = random Gaussian data, mean=0, std=1 if equal_var=True
        # or std=random number if equal_var=False,
        # to be compliant with H0, because should give a low value t
        Y = std * np.random.randn(n_meas_Y, n_vars)
        # add an offset to last variable, to reject H0
        # because should give a high value t
        Y[:, -1] = np.random.randn(n_meas_Y) + diff_mean

        t_pp, pvals = ppstats.permutation_ttest_ind(
            X, Y,
            n=100,
            with_replacement=False,
            side="one",
            equal_var=equal_var,
        )

        # assert t values with respect SciPy
        t_sp = np.zeros(n_vars)
        for v, (x, y) in enumerate(zip(X.T, Y.T)):
            t_sp[v] = stats.ttest_ind(x, y, equal_var=equal_var).statistic
        assert np.allclose(t_pp, t_sp, atol=1e-9), (
            "Statistics t should be equivalent to SciPy."
        )

        # the distribution is different only for the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha


###############################################################################


@pytest.mark.parametrize("with_replacement", (True, False))
@pytest.mark.parametrize("zero_method", ("pratt", "wilcox", "zsplit"))
@pytest.mark.parametrize("n_meas", [5, 7])
@pytest.mark.parametrize("n_vars", range(1, 5))
def test_permutation_wilcoxon_args(
    with_replacement,
    zero_method,
    n_meas,
    n_vars,
):
    """Check permutation_wilcoxon arguments.

    Parameters
    ----------
    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement.
    zero_method : {"pratt", "wilcox", "zsplit"}
        Method for zero-differences processing.
    n_meas : int
        Number of measurements in samples X and Y.
    n_vars : int
        Number of variables in samples X and Y.
    """
    if n_vars == 1:
        ppstats.permutation_wilcoxon(
            np.random.randn(n_meas),
            np.random.randn(n_meas),
            n=1,
            with_replacement=with_replacement,
        )

    stats, pvals = ppstats.permutation_wilcoxon(
        np.random.randn(n_meas, n_vars),
        np.random.randn(n_meas, n_vars),
        n=1,
        with_replacement=with_replacement,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_wilcoxon_errors(n_meas=5, n_vars=3):
    """Check permutation_wilcoxon errors."""
    with pytest.raises(ValueError):  # not same n_meas
        ppstats.permutation_wilcoxon(
            np.random.randn(n_meas, n_vars),
            np.random.randn(n_meas + 1, n_vars),
            n=1,
        )

    with pytest.raises(ValueError):  # not same n_vars
        ppstats.permutation_wilcoxon(
            np.random.randn(n_meas, n_vars),
            np.random.randn(n_meas, n_vars + 1),
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_wilcoxon(
            np.random.randn(n_meas, n_vars, 3),
            np.random.randn(n_meas, n_vars, 3),
            n=1,
        )


@pytest.mark.parametrize("zero_method", ("pratt", "wilcox", "zsplit"))
def test_permutation_wilcoxon_rates(
    zero_method,
    n_reps=50,
    n_meas=100,
    n_vars=10,
    diff_mean=2.,
    diff_std=1.,
    alpha=0.05,
):
    """Validate the permutation_wilcoxon function by generating random samples
    on which statistics and p-values are computed, and checking false positive
    and true positive rates.

    H0: difference between pairs follows a symmetric distribution around zero.
    H1: difference between pairs does not follow a symmetric distribution
    around zero.

    Parameters
    ----------
    zero_method : {"pratt", "wilcox", "zsplit"}
        Method for zero-differences processing.
    n_reps : int, default=50
        Number of repetitions.
    n_meas : int, default=100
        Number of measurements in samples X and Y.
    n_vars : int, default=10
        Number of variables in samples X and Y.
    diff_mean : float, default=2.0
        Value of the mean of the distribution in Y different from X.
    diff_std : float, default=1.0
        Value of the standard deviation of the distribution in Y different from
        X.
    alpha : float, default=0.05
        Type-I error use for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate
    """
    false_positives = []
    true_positives = []

    for i in range(n_reps):
        # X = random uniform data between 0 and 1
        X = np.random.rand(n_meas, n_vars)
        # Y = X + Gaussian noise, to be compliant with H0
        # ie, pairs difference follows a symmetric distribution around zero
        # because should give a high value T
        Y = X + 0.01 * np.random.randn(n_meas, n_vars)
        # add an offset to last variable, to reject H0
        # because should give a low value T
        Y[:, -1] = diff_std * np.random.randn(n_meas) + diff_mean

        T_pp, pvals = ppstats.permutation_wilcoxon(
            X, Y,
            n=100,
            with_replacement=False,
            zero_method=zero_method,
        )

        # assert T values with respect SciPy
        T_sp = np.zeros(n_vars)
        for v, (x, y) in enumerate(zip(X.T, Y.T)):
            T_sp[v] = stats.wilcoxon(x, y, zero_method=zero_method).statistic
        assert np.allclose(T_pp, T_sp, atol=1e-9), (
            "Statistics T should be equivalent to SciPy."
        )

        # the offset was only added to the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha


###############################################################################


@pytest.mark.parametrize("with_replacement", (True, False))
@pytest.mark.parametrize("n_meas_X", [10, 15, 20])
@pytest.mark.parametrize("n_meas_Y", [6, 7, 8])
@pytest.mark.parametrize("n_vars", range(1, 5))
def test_permutation_mannwhitneyu_args(
    with_replacement,
    n_meas_X,
    n_meas_Y,
    n_vars,
):
    """Check permutation_mannwhitneyu arguments.

    Parameters
    ----------
    with_replacement : bool
        Boolean to choose the bootstrap strategy: with replacement, or without
        replacement.
    n_meas_X : int
        Number of measurements in X.
    n_meas_Y : int
        Number of measurements in Y.
    n_vars : int
        Number of variables in X and Y.
    """
    if n_vars == 1:
        ppstats.permutation_mannwhitneyu(
            np.random.randn(n_meas_X),
            np.random.randn(n_meas_Y),
            n=1,
            with_replacement=with_replacement,
        )

    stats, pvals = ppstats.permutation_mannwhitneyu(
        np.random.randn(n_meas_X, n_vars),
        np.random.randn(n_meas_Y, n_vars),
        n=1,
        with_replacement=with_replacement,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_mannwhitneyu_errors(n_meas_X=3, n_meas_Y=4, n_vars=2):
    """Check permutation_mannwhitneyu errors."""
    with pytest.raises(ValueError):  # not same n_vars
        ppstats.permutation_mannwhitneyu(
            np.random.randn(n_meas_X, n_vars),
            np.random.randn(n_meas_Y, n_vars + 1),
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_mannwhitneyu(
            np.random.randn(n_meas_X, n_vars, 3),
            np.random.randn(n_meas_Y, n_vars, 3),
            n=1,
        )


def test_permutation_mannwhitneyu_rates(
    n_reps=50,
    n_meas_X=100,
    n_meas_Y=110,
    n_vars=10,
    diff_mean=1.,
    diff_std=1.,
    alpha=0.05,
):
    """Validate the permutation_mannwhitneyu function by generating random
    samples on which statistics and p-values are computed, and checking false
    positive and true positive rates.

    H0: samples are drawn from the same distribution, that may not be normal.

    Parameters
    ----------
    n_reps : int, default=50
        Number of repetitions.
    n_meas_X : int, default=110
        Number of measurements in X.
    n_meas_Y : int, default=100
        Number of measurements in Y.
    n_vars : int, default=10
        Number of variables in X and Y.
    diff_mean : float, default=1.0
        Value of the mean of the distribution in Y different from X.
    diff_std : float, default=1.0
        Value of the standard deviation of the distribution in Y different from
        X.
    alpha : float, default=0.05
        Type-I error use for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate.
    """
    false_positives = []
    true_positives = []

    for i in range(n_reps):
        # X = random Gaussian data, mean=0, std=1
        X = np.random.randn(n_meas_X, n_vars)
        # Y = random Gaussian data, mean=0, std=1, to be compliant with H0
        # because should give a high value U
        Y = np.random.randn(n_meas_Y, n_vars)
        # add an offset to last variable, to reject H0
        # because should give a low value U
        Y[:, -1] = diff_std * np.random.randn(n_meas_Y) + diff_mean

        U_pp, pvals = ppstats.permutation_mannwhitneyu(
            X, Y,
            n=100,
            with_replacement=False,
        )

        # TODO: assert U values with respect SciPy
        # NOT DONE, because scipy implementation is really "different" ...
#        U_sp = np.zeros(n_vars)
#        for v, (x, y) in enumerate(zip(X.T, Y.T)):
#            U_sp[v] = stats.mannwhitneyu(x, y, alternative="less").statistic
#        assert np.allclose(U_pp, U_sp, atol=1e-9), (
#            "Statistics U should be equivalent to SciPy."
#        )

        # the distribution is different only for the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha


###############################################################################


@pytest.mark.parametrize("list_meas", [[5, 7, 6, 4], [10, 2, 5]])
@pytest.mark.parametrize("n_vars", range(1, 4))
def test_permutation_f_oneway_args(list_meas, n_vars):
    """Check permutation_f_oneway arguments.

    Parameters
    ----------
    list_meas : list of int
        List of number of measurements of each sample.
        Its length defines the number of samples / groups.
    n_vars : int
        Number of variables in samples.
    """
    if n_vars == 1:
        ppstats.permutation_f_oneway(
            *[np.random.randn(n_meas) for n_meas in list_meas],
            n=1,
        )

    stats, pvals = ppstats.permutation_f_oneway(
        *[np.random.randn(n_meas, n_vars) for n_meas in list_meas],
        n=1,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_f_oneway_errors(list_meas=[2, 4, 6, 3], n_vars=3):
    """Check permutation_f_oneway errors."""
    with pytest.raises(ValueError):  # only one group
        ppstats.permutation_f_oneway(
            *[np.random.randn(list_meas[0], n_vars)],
            n=1,
        )

    with pytest.raises(ValueError):  # not same n_vars
        list_vars = np.random.randint(1, 10, size=len(list_meas))
        ppstats.permutation_f_oneway(
            *[
                np.random.randn(n_meas, n_vars_)
                for n_meas, n_vars_ in zip(list_meas, list_vars)
            ],
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_f_oneway(
            *[np.random.randn(n_meas, n_vars, 3) for n_meas in list_meas],
            n=1,
        )


def test_permutation_f_oneway_rates(
    n_reps=50,
    list_meas=[80, 100, 50, 70],
    n_vars=3,
    diff_mean=1.,
    diff_std=1.,
    alpha=0.05,
):
    """Validate the permutation_f_oneway function by generating random samples
    on which statistics and p-values are computed, and checking false positive
    and true positive rates.

    H0: means of Gaussian samples are all equal.

    Parameters
    ----------
    n_reps : int, default=50
        Number of repetitions.
    list_meas : list of int, default=[80, 100, 50, 70]
        List of number of measurements of each sample.
        Its length defines the number of samples / groups.
    n_vars : int, default=3
        Number of variables in samples.
    diff_mean : float, default=1.0
        Value of the mean of the distribution in the last variable different
        from others.
    diff_std : float, default=1.0
        Value of the standard deviation of the distribution in the last
        variable different from others.
    alpha : float, default=0.05
        Type-I error use for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate.
    """
    false_positives = []
    true_positives = []

    for i in range(n_reps):
        args = []
        for j, n_meas in enumerate(list_meas):
            # all samples = random Gaussian data, mean=0, std=1
            # to be compliant with H0, because should give a low value F
            X = np.random.randn(n_meas, n_vars)
            if j == len(list_meas)-1:
                # add an offset to last sample of the last variable,
                # to reject H0, because should give a high value F
                X[:, -1] = diff_std * np.random.randn(n_meas) + diff_mean
            args.append(X)

        F_pp, pvals = ppstats.permutation_f_oneway(*args, n=500)

        # assert F values with respect SciPy
        F_sp = np.zeros(n_vars)
        for v in range(n_vars):
            args_ = [arg[:, v] for arg in args]
            F_sp[v] = stats.f_oneway(*args_).statistic
        assert np.allclose(F_pp, F_sp, atol=1e-9), (
            "Statistics F should be equivalent to SciPy."
        )

        # the distribution is different only for the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha


###############################################################################


@pytest.mark.parametrize("list_meas", [[4, 3, 5, 7], [10, 2, 5]])
@pytest.mark.parametrize("n_vars", range(1, 4))
def test_permutation_kruskal_args(list_meas, n_vars):
    """Check permutation_kruskal arguments.

    Parameters
    ----------
    list_meas : list of int
        List of number of measurements of each sample.
        Its length defines the number of samples / groups.
    n_vars : int
        Number of variables in samples.
    """
    if n_vars == 1:
        ppstats.permutation_kruskal(
            *[np.random.randn(n_meas) for n_meas in list_meas],
            n=1,
        )

    stats, pvals = ppstats.permutation_kruskal(
        *[np.random.randn(n_meas, n_vars) for n_meas in list_meas],
        n=1,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_kruskal_errors(list_meas=[4, 3, 5, 3], n_vars=3):
    """Check permutation_kruskal errors."""
    with pytest.raises(ValueError):  # only one group
        ppstats.permutation_kruskal(
            *[np.random.randn(list_meas[0], n_vars)],
            n=1,
        )

    with pytest.raises(ValueError):  # not same n_vars
        list_vars = np.random.randint(1, 10, size=len(list_meas))
        ppstats.permutation_kruskal(
            *[
                np.random.randn(n_meas, n_vars_)
                for n_meas, n_vars_ in zip(list_meas, list_vars)
            ],
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_kruskal(
            *[np.random.randn(n_meas, n_vars, 3) for n_meas in list_meas],
            n=1,
        )


def test_permutation_kruskal_rates(
    n_reps=50,
    list_meas=[80, 100, 50, 70],
    n_vars=3,
    diff_mean=1.,
    diff_std=1.,
    alpha=0.05,
):
    """Validate the permutation_kruskal function by generating random samples
    on which statistics and p-values are computed, and checking false positive
    and true positive rates.

    H0: medians of samples are all equal.

    Parameters
    ----------
    n_reps : int, default=50
        Number of repetitions.
    list_meas : list of int, default=[80, 100, 50, 70]
        List of number of measurements of each sample.
        Its length defines the number of samples / groups.
    n_vars : int, default=3
        Number of variables in samples.
    diff_mean : float, default=1.0
        Value of the mean of the distribution in the last variable different
        from others.
    diff_std : float, default=1.0
        Value of the standard deviation of the distribution in the last
        variable different from others.
    alpha : float, default=0.05
        Type-I error use for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate.
    """
    false_positives = []
    true_positives = []

    for i in range(n_reps):
        args = []
        for j, n_meas in enumerate(list_meas):
            # all samples = random Gaussian data, mean=0, std=1
            # to be compliant with H0, because should give a low value H
            X = np.random.randn(n_meas, n_vars)
            if j == len(list_meas)-1:
                # add an offset to last sample of the last variable,
                # to reject H0, because should give a high value H
                X[:, -1] = diff_std * np.random.randn(n_meas) + diff_mean
            args.append(X)

        H_pp, pvals = ppstats.permutation_kruskal(*args, n=500)

        # assert H values with respect SciPy
        H_sp = np.zeros(n_vars)
        for v in range(n_vars):
            args_ = [arg[:, v] for arg in args]
            H_sp[v] = stats.kruskal(*args_).statistic
        assert np.allclose(H_pp, H_sp, atol=1e-9), (
            "Statistics H should be equivalent to SciPy."
        )

        # the distribution is different only for the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha


###############################################################################


@pytest.mark.parametrize("n_meas", range(1, 5))
@pytest.mark.parametrize("n_vars", range(1, 5))
@pytest.mark.parametrize("n_groups", range(3, 5))
def test_permutation_friedmanchisquare_args(n_meas, n_vars, n_groups):
    """Check permutation_friedmanchisquare arguments.

    Parameters
    ----------
    n_meas : int
        Number of measurements in samples.
    n_vars : int
        Number of variables in samples.
    n_groups : int
        Number of samples / groups.
    """
    if n_vars == 1:
        ppstats.permutation_friedmanchisquare(
            *[np.random.randn(n_meas) for _ in range(n_groups)],
            n=1,
        )

    stats, pvals = ppstats.permutation_friedmanchisquare(
        *[np.random.randn(n_meas, n_vars) for _ in range(n_groups)],
        n=1,
    )
    assert stats.shape == (n_vars,)
    assert pvals.shape == (n_vars,)


def test_permutation_friedmanchisquare_errors(n_meas=3, n_vars=2, n_groups=4):
    """Check permutation_friedmanchisquare errors."""
    with pytest.raises(ValueError):  # only two groups
        ppstats.permutation_friedmanchisquare(
            *[np.random.randn(n_meas) for _ in range(2)],
            n=1,
        )

    with pytest.raises(ValueError):  # not same n_meas
        list_meas = np.random.randint(1, 10, size=n_groups)
        ppstats.permutation_friedmanchisquare(
            *[np.random.randn(n_meas_, n_vars) for n_meas_ in list_meas],
            n=1,
        )

    with pytest.raises(ValueError):  # not same n_vars
        list_vars = np.random.randint(1, 10, size=n_groups)
        ppstats.permutation_friedmanchisquare(
            *[np.random.randn(n_meas, n_vars_) for n_vars_ in list_vars],
            n=1,
        )

    with pytest.raises(ValueError):  # more than 2 dims
        ppstats.permutation_friedmanchisquare(
            *[np.random.randn(n_meas, n_vars, 3) for _ in range(n_groups)],
            n=1,
        )


def test_permutation_friedmanchisquare_rates(
    n_reps=50,
    n_meas=10,
    n_vars=3,
    n_groups=5,
    diff_mean=3.,
    diff_std=0.5,
    alpha=0.05,
):
    """Validate the permutation_friedmanchisquare function by generating random
    samples on which statistics and p-values are computed, and checking false
    positive and true positive rates.

    H0: medians of samples are all equal.

    Parameters
    ----------
    n_reps : int, default=50
        Number of repetitions.
    n_meas : int, default=10
        Number of measurements in samples.
    n_vars : int, default=3
        Number of variables in samples.
    n_groups : int, default=5
        Number of samples / groups.
    diff_mean : float, default=3.0
        Value of the mean of the distribution in the last variable different
        from others.
    diff_std : float, default=0.5
        Value of the standard deviation of the distribution in the last
        variable different from others.
    alpha : float, default=0.05
        Type-I error use for the generation of false and true positive rates.

    Returns
    -------
    fpr : float
        False positive rate.
    tpr : float
        True positive rate.
    """
    false_positives = []
    true_positives = []

    for i in range(n_reps):
        args = []
        for j in range(n_groups):
            # all samples = random Gaussian data, mean=0, std=1
            # to be compliant with H0, because should give a low value chi2
            X = np.random.randn(n_meas, n_vars)
            if j == n_groups-1:
                # add an offset to last sample of the last variable,
                # to reject H0, because should give a high value chi2
                X[:, -1] = diff_std * np.random.randn(n_meas) + diff_mean
            args.append(X)

        chi2_pp, pvals = ppstats.permutation_friedmanchisquare(*args, n=500)

        # assert chi2 values with respect SciPy
        chi2_sp = np.zeros(n_vars)
        for v in range(n_vars):
            args_ = [arg[:, v] for arg in args]
            chi2_sp[v] = stats.friedmanchisquare(*args_).statistic
        assert np.allclose(chi2_pp, chi2_sp, atol=1e-9), (
            "Statistics chi2 should be equivalent to SciPy."
        )

        # the distribution is different only for the last index
        false_positives.append(sum([int(p < alpha) for p in pvals[:-1]]))
        true_positives.append(int(pvals[-1] < alpha))

    # assert false positive and true positive rates
    fpr = sum(false_positives) / ((n_vars - 1) * n_reps)
    tpr = sum(true_positives) / n_reps
    assert fpr < alpha
    assert tpr > 1. - alpha
