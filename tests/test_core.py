""" Tests for module core.

To execute tests:
>>> py.test -k test_core
"""

import pytest
from itertools import combinations
import pypermut.core as core


@pytest.mark.parametrize("n", range(2, 9))
def test_permutations_measurements(n):
    """ Test permutations along measurements:
        uniqueness and null permutation.

    Parameter
    ---------
    n : int
        Length of samples that will be tested. Do not use values over 8, since
        it would take too long to permute completely.
    """
    assert n > 1, 'Need at least 2 values.'

    max_perms = core.count_permutation_measurements(n)
    # Get all permutations
    perms = []
    for p in range(max_perms):
        perm = core.get_permutation_measurements(n, p)
        perms.append(perm)

    # Uniqueness: there should be exactly max_perms permutations,
    # and using a set will help determine if there are repeated results.
    permutations = set(tuple(perm) for perm in perms)
    assert len(permutations) == max_perms, 'Generated permutations are not unique.'

    # Null permutation
    assert perms[0] == list(range(n)), 'First permutation is not the null permutation.'


@pytest.mark.parametrize("n", range(2, 15))
def test_permutations_paired_samples(n):
    """ Test permutations for paired samples:
        uniqueness and null/full permutations.

    Parameter
    ---------
    n : int
        Length of samples that will be tested.
    """
    assert n > 1, 'Need at least 2 values.'

    max_perms = core.count_permutations_paired_samples(2, n)
    # Get all permutations
    perms = []
    for p in range(max_perms):
        perm = core.get_permutation_2_paired_samples(n, p)
        perms.append(perm)

    # Uniqueness: there should be exactly max_perms permutations,
    # and using a set will help determine if there are repeated results.
    permutations = set(tuple(map(tuple, perm)) for perm in perms)
    assert len(permutations) == max_perms, 'Generated permutations are not unique.'

    # Null and full permutations
    assert all(v == 1 for v in perms[0]), 'First permutation is not the null permutation.'
    assert all(v == -1 for v in perms[-1]), 'Last permutation is not the full permutation.'


@pytest.mark.parametrize("n1", range(2, 14))
@pytest.mark.parametrize("n2", range(2, 10))
def test_permutations_unpaired_samples(n1, n2):
    """ Test permutations for unpaired samples:
        uniqueness and null permutation.

    Parameters
    ----------
    n1 : int
        Length of first sample that will be tested.

    n2 : int
        Length of second sample that will be tested.
    """
    assert n1 > 1, 'Need at least 2 values.'
    assert n2 > 1, 'Need at least 2 values.'

    max_perms = core.count_permutations_unpaired_samples([n1, n2])
    # Get all combinations
    combs = list(combinations(range(n1 + n2), n1))
    # Get all permutations
    perms = []
    for p in range(max_perms):
        perm = core.get_permutation_unpaired_samples([n1, n2], list(combs[p]))
        perms.append(perm)

    # Uniqueness: there should be exactly max_perms permutations,
    # and using a set will help determine if there are repeated results.
    permutations = set(tuple(perm) for perm in perms)
    assert len(permutations) == max_perms, 'Generated permutations are not unique.'

    # Null permutation
    assert perms[0] == list(range(n1 + n2)), 'First permutation is not the null permutation.'
