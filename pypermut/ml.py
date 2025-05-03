"""Permutations for machine learning.

Random state can be fixed before calling permutation tests,
using np.random.seed().
"""

import numpy as np
from scipy.stats import percentileofscore

from .misc import perc_to_pval


def permutation_metric(y_true, y_score, func, *, n=10000, side="right"):
    """Permutation test for machine learning metric.

    This function performs a permutation test on any metric based on the
    predictions of a model. It permutes labels and predictions to obtain a
    p-value for any machine learning metrics:

    * the Area Under the Receiver Operating Characteristic (AUROC) curve,
    * the Area Under the Precision-Recall (AUPR) curve,
    * the negative log-likelihood (log-loss),
    * etc.

    Parameters
    ----------
    y_true : array_like, shape (n_samples, n_classes)
        True binary labels, with first dimension representing the sample
        dimension and with second dimension representing the different classes.
    y_score : array_like, shape (n_samples, n_classes)
        Scores of prediction, same dimensions as y_true. Scores can be
        probabilities or labels.
    func : callable
        Function to compute the metric, with signature `func(y_true, y_score)`.
    n : int, default=10000
        Number of permutations for the permutation test.
    side : {"left", "two", "right"}, default="right"
        Side of the test:

        * "left" for a left-sided test,
        * "two" or "double" for a two-sided test,
        * "right" for a right-sided test.

    Returns
    -------
    m : float
        The value of the metric.
    pval : float
        The p-value associated to the metric.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """  # noqa
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            "Inputs y_true and y_score do not have compatible dimensions: "
            "y_true is of dimension {} while y_score is {}."
            .format(y_true.shape, y_score.shape)
        )
    n_samples = y_true.shape[0]

    # under the null hypothesis, sample the metric distribution
    null_dist = np.empty(n, dtype=float)
    for p in range(n):
        permuted_indices = np.random.permutation(n_samples)
        null_dist[p] = func(y_true[permuted_indices], y_score)

    # compute the real metric
    m = func(y_true, y_score)
    perc = percentileofscore(null_dist, m, kind="strict")
    pval = perc_to_pval(perc, side)

    return m, pval


def standard_error_auroc(y_labels, y_probas, auroc):
    """Standard error of AUROC.

    This function computes the standard error of the area under the
    receiver operating characteristic (ROC) curve, ie AUROC.

    Parameters
    ----------
    y_labels : array_like, shape (n_samples,)
        True binary labels.
    y_probas : array_like, shape (n_samples,)
        Probabilities of predicted labels, same dimension as y_labels.
    auroc : float
        AUROC value, between 0 and 1.

    Returns
    -------
    se : float
        Standard error of the area underneath empirical ROC curve.

    References
    ----------
    .. [1] Hanley & McNeil, "The meaning and use of the area under a receiver
           operating characteristic (ROC) curve", Radiology, 1982.
    """
    y_labels = np.asarray(y_labels)
    if y_labels.ndim != 1:
        raise ValueError("Inputs must have only one dimension")
    y_probas = np.asarray(y_probas)
    if y_labels.shape != y_probas.shape:
        raise ValueError(
            "Inputs y_labels and y_probas do not have compatible dimensions: "
            "y_labels is of dimension {} while y_probas is {}."
            .format(y_labels.shape, y_probas.shape)
        )
    if not 0 <= auroc <= 1:
        raise ValueError(f"Input auroc={auroc} must be included in [0, 1].")

    X_A = y_probas[y_labels == 1]
    n_A = X_A.shape[0]
    X_N = y_probas[y_labels == 0]
    n_N = X_N.shape[0]

    # Eq(2)
    Q1 = auroc / (2 - auroc)
    Q2 = 2 * auroc**2 / (1 + auroc)
    # Eq(1)
    se = np.sqrt((auroc * (1-auroc) + (n_A-1) * (Q1-auroc**2)
                 + (n_N-1) * (Q2-auroc**2)) / (n_A*n_N))

    return se
