"""Tests for module core.

To execute tests:
>>> pytest -k test_core
"""

from itertools import combinations

import numpy as np
import pytest

import pypermut.core as core
import pypermut.helpers as helpers
import pypermut.mstats as mstats


###############################################################################


@pytest.mark.parametrize("n_permutations", [5, 7, "all"])
@pytest.mark.parametrize("with_replacement", [True, False])
@pytest.mark.parametrize("stat_func", [mstats.pearsonr, mstats.spearmanr])
@pytest.mark.parametrize("side", ["one", "two"])
def test_permute_measurements(
    n_permutations,
    with_replacement,
    stat_func,
    side,
    n_meas=5,
    n_vars=3
):
    """Test permutation test of two samples along measurement."""
    null_dist = core.permute_measurements(
        Y=np.random.random((n_meas, n_vars)),
        x=np.random.random((n_meas)),
        n_permutations=n_permutations,
        with_replacement=with_replacement,
        stat_func=stat_func,
        var_func=np.max,
        side=side,
    )

    if n_permutations == "all":
        n_permutations = core.count_permutation_measurements(n_meas)
    assert null_dist.shape == (n_permutations,)


@pytest.mark.parametrize("n_permutations", [5, 7, "all"])
@pytest.mark.parametrize("with_replacement", [True, False])
@pytest.mark.parametrize("stat_func", [mstats.studentt_rel, mstats.wilcoxon])
@pytest.mark.parametrize("var_func", [np.max, np.min])
@pytest.mark.parametrize("side", ["one", "two"])
def test_permute_paired_samples(
    n_permutations,
    with_replacement,
    stat_func,
    var_func,
    side,
    n_meas=5,
    n_vars=3
):
    """Test permutation test of paired samples."""
    kwargs = {}
    if stat_func is mstats.wilcoxon:
        kwargs["zero_method"] = "wilcox"

    null_dist = core.permute_paired_samples(
        X=np.random.random((n_meas, n_vars)),
        Y=np.random.random((n_meas, n_vars)),
        n_permutations=n_permutations,
        with_replacement=with_replacement,
        stat_func=stat_func,
        var_func=var_func,
        side=side,
        **kwargs,
    )

    if n_permutations == "all":
        n_permutations = core.count_permutations_paired_samples(2, n_meas)
    assert null_dist.shape == (n_permutations,)


@pytest.mark.parametrize("n_permutations", [3, 4, "all"])
@pytest.mark.parametrize("with_replacement", [True, False])
@pytest.mark.parametrize("stat_func", [
    mstats.studentt_ind,
    mstats.welcht_ind,
    mstats.mannwhitneyu,
    mstats.f_oneway,
    mstats.kruskal
])
@pytest.mark.parametrize("var_func", [np.max, np.min])
@pytest.mark.parametrize("side", ["one", "two"])
@pytest.mark.parametrize("n_samples", [2, 3])
def test_permute_unpaired_samples(
    n_permutations,
    with_replacement,
    stat_func,
    var_func,
    side,
    n_samples,
    n_meas=3,
    n_vars=2
):
    """Test permutation test of unpaired samples."""
    X = [np.random.random((n_meas, n_vars)) for _ in range(n_samples)]

    list_meas = helpers.get_list_meas(X)
    n_perms_max = core.count_permutations_unpaired_samples(list_meas)
    perms, n_perms, with_replacement = helpers.check_permutations(
        n_perms_requested=n_permutations,
        n_perms_max=n_perms_max,
        with_replacement=with_replacement,
    )

    if n_samples > 2 and not with_replacement:
        return

    null_dist = core.permute_unpaired_samples(
        X,
        n_permutations=n_permutations,
        with_replacement=with_replacement,
        stat_func=stat_func,
        var_func=var_func,
        side=side,
    )

    if n_permutations == "all":
        n_permutations = n_perms_max
    assert null_dist.shape == (n_permutations,)


###############################################################################


@pytest.mark.parametrize("n", range(2, 9))
def test_get_permutation_measurements(n):
    """Test permutations along measurements.

    Parameter
    ---------
    n : int
        Length of samples that will be tested. Do not use values over 8, since
        it would take too long to permute completely.
    """
    assert n > 1, "Need at least 2 values."

    max_perms = core.count_permutation_measurements(n)
    # get all permutations
    perms = []
    for p in range(max_perms):
        perm = core.get_permutation_measurements(n, p)
        perms.append(perm)

    # uniqueness: there should be exactly max_perms permutations,
    # and using a set will help determine if there are repeated results
    permutations = set(tuple(perm) for perm in perms)
    assert len(permutations) == max_perms, "Generated permutations are not unique."

    # null and reverse permutations
    inds = list(range(n))
    assert perms[0] == inds, "First permutation is not the null permutation."
    assert perms[-1] == list(reversed(inds)), "Last permutation is not the reverse permutation."


@pytest.mark.parametrize("n", range(2, 15))
def test_get_permutation_2_paired_samples(n):
    """Test permutations for two paired samples.

    Parameter
    ---------
    n : int
        Length of samples that will be tested.
    """
    assert n > 1, "Need at least 2 values."

    max_perms = core.count_permutations_paired_samples(2, n)
    # get all permutations
    perms = []
    for p in range(max_perms):
        perm = core.get_permutation_2_paired_samples(n, p)
        perms.append(perm)

    # uniqueness: there should be exactly max_perms permutations,
    # and using a set will help determine if there are repeated results
    permutations = set(tuple(map(tuple, perm)) for perm in perms)
    assert len(permutations) == max_perms, "Generated permutations are not unique."

    # null and full permutations
    assert all(v == 1 for v in perms[0]), "First permutation is not the null permutation."
    assert all(v == -1 for v in perms[-1]), "Last permutation is not the full permutation."


@pytest.mark.parametrize("n1", range(2, 14))
@pytest.mark.parametrize("n2", range(2, 10))
def test_get_permutation_unpaired_samples(n1, n2):
    """Test permutations for unpaired samples.

    Parameters
    ----------
    n1 : int
        Length of first sample that will be tested.
    n2 : int
        Length of second sample that will be tested.
    """
    assert n1 > 1, "Need at least 2 values."
    assert n2 > 1, "Need at least 2 values."

    max_perms = core.count_permutations_unpaired_samples([n1, n2])
    # get all combinations
    combs = list(combinations(range(n1 + n2), n1))
    # get all permutations
    perms = []
    for p in range(max_perms):
        perm = core.get_permutation_unpaired_samples([n1, n2], list(combs[p]))
        perms.append(perm)

    # uniqueness: there should be exactly max_perms permutations,
    # and using a set will help determine if there are repeated results
    permutations = set(tuple(perm) for perm in perms)
    assert len(permutations) == max_perms, "Generated permutations are not unique."

    # null permutation
    assert perms[0] == list(range(n1 + n2)), "First permutation is not the null permutation."
