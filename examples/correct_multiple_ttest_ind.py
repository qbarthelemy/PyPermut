"""
===============================================================================
Correct multiple Student's t-tests for independent samples
===============================================================================

Correct multiple Student's t-tests for trivariate independent samples,
comparing three methods: Bonferroni, permutations, and Hotelling.
"""
# Author: Quentin Barthélemy

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, f

from pypermut.misc import print_results, print_pvals, pvals_to_stars
from pypermut.stats import permutation_ttest_ind


###############################################################################
# Multivariate data
# -----------------

# Artificial trivariate Gaussian samples, with a shift in the third variable
n_meas, n_vars = 50, 3
np.random.seed(42)
X = np.random.randn(n_meas, n_vars)
Y = np.random.randn(n_meas, n_vars)
Y[:, 2] -= 0.8
vlabels = ["var{}".format(v) for v in range(n_vars)]

# Plot trivariate samples
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d", title="3D visualization",
                      xlabel=vlabels[0], ylabel=vlabels[1], zlabel=vlabels[2])
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], label="1st sample")
ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], marker="^", label="2nd sample")
plt.legend(loc="center right")
ax2 = fig.add_subplot(122, projection="3d", title="2D projection",
                      xlabel=vlabels[0], ylabel=vlabels[1], zlabel=vlabels[2])
for D, c in zip([X, Y], ["C0o", "C1^"]):
    ax2.plot(D[:, 0], D[:, 1], c, zdir="z", zs=ax1.get_zlim()[0], ms=4, alpha=0.5)
    ax2.plot(D[:, 0], D[:, 2], c, zdir="y", zs=ax1.get_ylim()[-1], ms=4, alpha=0.5)
    ax2.plot(D[:, 1], D[:, 2], c, zdir="x", zs=ax1.get_xlim()[0], ms=4, alpha=0.5)
plt.show()


###############################################################################
# Student's t-tests for independent samples
# -----------------------------------------

# Classical tests
t_class = [ttest_ind(x, y, alternative="greater") for x, y in zip(X.T, Y.T)]
print("Classical t-tests:")
print_results(t_class, vlabels, "t")

# Bonferroni correction, to control the family-wise error rate for multiple
# tests
p_corrected = np.array(t_class)[:, 1] * n_vars
print("Classical t-tests, after Bonferroni correction:")
print_pvals(p_corrected, vlabels)

# Permutated tests
n_perm = 10000
t_perm, p_perm = permutation_ttest_ind(X, Y, n=n_perm, side="one")
print("\nPermutation t-tests:")
print_results(np.c_[t_perm, p_perm], vlabels, 't')

# Classical t-tests corrected by Bonferroni detect a trend in the difference
# between the third variables of samples, while permutation tests find a
# significant difference.


###############################################################################
# Hotelling's two-sample t-squared test, for multivariate samples
# ---------------------------------------------------------------

# Hotelling's two-sample t-squared test is the multivariate generalization of
# the Student's t-test [1].

def hotelling(X, Y):
    n_meas_X, n_meas_Y = X.shape[0], Y.shape[0]
    n_vars = X.shape[1]
    assert Y.shape[1] == n_vars

    n_meas = n_meas_X + n_meas_Y
    diff = np.mean(X, axis=0) - np.mean(Y, axis=0)
    Cov_X = np.cov(X, rowvar=False)
    Cov_Y = np.cov(Y, rowvar=False)
    Cov_pooled = ((n_meas_X-1) * Cov_X + (n_meas_Y-1) * Cov_Y) / (n_meas-2)
    Cov_pooled_inv = np.linalg.pinv(Cov_pooled)
    tsq = (n_meas_X*n_meas_Y) / n_meas * diff.T @ Cov_pooled_inv @ diff

    Fstat = tsq * (n_meas - n_vars - 1) / ((n_meas-2) * n_vars)
    Fdist = f(n_vars, n_meas - 1 - n_vars)
    pval = 1 - Fdist.cdf(Fstat)

    return tsq, pval

tsq, p_multiv = hotelling(X, Y)
print("\nHotelling t-squared test:\n t^2={:.2f}, p={:.3e} ({})"
      .format(tsq, p_multiv, pvals_to_stars(p_multiv)))

# Hotelling's test detects a trend in the difference between samples too, and
# is not able to say on which dimension.


###############################################################################
# Reference
# ---------
# [1] Hotelling, "The Generalization of Student’s Ratio", Ann Math Statist,
# 1931.
