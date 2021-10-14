""" Tests for module ml. 

To execute tests:
>>> py.test -k test_ml
"""

import pytest
import numpy as np
import pypermut.ml as ppml

np.random.seed(17)


def permutation_metric_func(y_true, y_score):
    """ Simple function for test purpose """
    return np.mean(y_true - y_score)


@pytest.mark.parametrize("side", ('left', 'two', 'right'))
def test_permutation_metric_args(side, n_samples=10, n_classes=3):
    """ This function checks permutation_metric arguments.

    Parameters
    ----------
    side : string
        Side of the test:

        * 'left' for a left-sided test,
        * 'two' or 'double' for a two-sided test,
        * 'right' for a right-sided test.

    n_samples : int, optional
        Number of samples.

    n_classes : int, optional
        Number of classes.
    """
    y_true = np.random.randn(n_samples, n_classes)
    y_score = np.random.randn(n_samples, n_classes)

    m, p = ppml.permutation_metric(y_true,
                                   y_score,
                                   permutation_metric_func,
                                   n=1,
                                   side=side)


def test_permutation_metric_errors(n_samples=10, n_classes=3):
    """ This function checks permutation_metric errors.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples.

    n_classes : int, optional
        Number of classes.
    """
    with pytest.raises(ValueError):  # not same n_samples
        ppml.permutation_metric(np.random.randn(n_samples, n_classes),
                                np.random.randn(n_samples + 1, n_classes),
                                permutation_metric_func,
                                n=1)

    with pytest.raises(ValueError):  # not same n_classes
        ppml.permutation_metric(np.random.randn(n_samples, n_classes),
                                np.random.randn(n_samples, n_classes + 1),
                                permutation_metric_func,
                                n=1)

    with pytest.raises(ValueError):  # unknown side
        ppml.permutation_metric(np.random.randn(n_samples, n_classes),
                                np.random.randn(n_samples, n_classes),
                                permutation_metric_func,
                                n=1,
                                side='abc')
