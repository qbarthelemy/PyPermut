""" Miscellaneous functions. """

import numpy as np
from matplotlib import pyplot as plt


def perc_to_pval(perc, side):
    """ This function transforms a percentile into a p-value,
    depending on the side of the test.

    Parameters
    ----------
    perc : float
        Percentile of the observed statistic, in [0, 100].

    side : string
        Side of the test:

        * 'left', 'lower' or 'less', for a left-sided test;
        * 'two', 'double' or 'two-sided', for a two-sided test;
        * 'right', 'upper' or 'greater', for a right-sided test.

    Returns
    -------
    pval : float
        The p-value associated to the stat.
    """
    if not 0 <= perc <= 100:
        raise ValueError('Input percentile="{}" must be included in [0, 100].'
                         .format(perc))

    if side in ['left', 'lower', 'less']:
        pval = perc / 100
    elif side in ['two', 'double', 'two-sided']:
        pval = 2 * min(perc / 100, (100 - perc) / 100)
    elif side in ['right', 'upper', 'greater']:
        pval = (100 - perc) / 100
    else:
        raise ValueError('Invalid value for side="{}".'.format(side))

    return pval


def pvals_to_stars(p_vals, *,
                   p_thresholds=[0.001, 0.01, 0.05, 0.1, 1],
                   p_notations=['***', '**', '*', '+', '', '']):
    """ This function transforms p-values into the star-based notation.

    Parameters
    ----------
    p_vals : array of float
        Array of p-values.

    p_thresholds : list of float (default [0.001, 0.01, 0.05, 0.1, 1])
        List of p-value thresholds.

    p_notations : list of string (default ['***', '**', '*', '+', '', ''])
        List of star-based notations, of length(p_thresholds) + 1.

    Returns
    -------
    stars : array of string
        Array of stars.
    """
    if len(p_thresholds) != len(p_notations) - 1:
        raise ValueError(
            'Input p_thresholds must have one element less than p_thresholds: '
            'p_thresholds is of length {} while p_notations is {}.'
            .format(len(p_thresholds), len(p_notations)))

    notations = np.array(p_notations)
    stars = notations[np.digitize(p_vals, p_thresholds, right=True)]

    return stars


def print_results(results, r_labels, stat_label):
    """ This function prints results of several tests: statistic value and the
    p-value.

    Parameters
    ----------
    results : list of list
        List of test results, ie a list containing the statistic value and the
        p-value.

    r_labels : list of string
        List of labels, one for each test.

    stat_label : string
        String denoting the statistic name.
    """
    if len(results) != len(r_labels):
        raise ValueError(
            'Inputs results and r_labels do not have compatible lengths: '
            'results is of length {} while r_labels is {}.'
            .format(len(results), len(r_labels)))

    for r_label, res in zip(r_labels, results):
        print(' {} : {}={:.3f}, p={:.3e} ({})'
              .format(r_label, stat_label, res[0], res[1],
                      pvals_to_stars(res[1]))
        )


def print_pvals(pvals, r_labels):
    """ This function prints p-values of several tests.

    Parameters
    ----------
    pvals : list of float
        List of p-values.

    r_labels : list of string
        List of labels, one for each test.
    """
    if len(pvals) != len(r_labels):
        raise ValueError(
            'Inputs pvals and r_labels do not have compatible lengths: '
            'pvals is of length {} while r_labels is {}.'
            .format(len(pvals), len(r_labels)))

    for r_label, p in zip(r_labels, pvals):
        print(' {} : p={:.3e} ({})'.format(r_label, p, pvals_to_stars(p)))


def plot_pairwise_results(ax, names, pvals):
    """ This function plots the significance of pairwise tests in a square
    matrix.

    Parameters
    ----------    
    ax : matplotlib axes
        Axis of the figure.

    names : list of string, length(n_names)
        Names of tested pairs.

    pvals : list of float, length(n_names * (n_names - 1) / 2)
        List of p-values for all pairwise tests.

    Returns
    -------
    ax : matplotlib axes
        Axis of the figure.
    """
    n_names, n_pvals = len(names), len(pvals)
    if  n_pvals != n_names * (n_names - 1) // 2:
        raise ValueError(
            'Inputs names and pvals do not have compatible lengths: '
            'names is of length {} while pvals is {}.'
            .format(n_names, n_pvals))

    # prepare pairwise indices
    unravel_inds = []
    for i in np.arange(0, n_names - 1):
        for j in np.arange(i + 1, n_names):
            unravel_inds.append([i, j])

    # plot square matrix
    ax.matshow(np.zeros((n_names, n_names)), cmap=plt.cm.Blues)
    tick_marks = np.arange(n_names) + 0.5
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([])
    for i, name in enumerate(names):
        ax.text(i, i, name, horizontalalignment='center',
                verticalalignment='top')
    for i, pval in enumerate(pvals):
        ax.text(unravel_inds[i][0], unravel_inds[i][1], pvals_to_stars(pval),
                horizontalalignment='center', verticalalignment='top')

    return ax
