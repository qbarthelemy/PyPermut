"""
===============================================================================
Correct Pearson and Spearman correlations on Anscombe's data
===============================================================================

Compare classical and permutated tests on Anscombe's data sets [1],
for Pearson and Spearman correlations after correction for multiple tests.
"""
# Author: Quentin Barthélemy

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from pypermut.misc import print_results, print_pvals
from pypermut.stats import permutation_corr


###############################################################################
# Multivariate data
# -----------------

# Anscombe's trio: only the first three data sets of the Anscombe's quartet,
# because the last set does not share the same values x.
x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
Y = np.array([
    [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84,4.82, 5.68],
    [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
    [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
]).T

# Anscombe's anti-trio: to highlight the difference between classical and
# permutated tests, the number of variables is artificially increased.
Y = np.c_[Y, -Y]
vlabels = ["y{}".format(v + 1) for v in range(Y.shape[1])]

# Plot Anscombe's trio and anti-trio
fig, ax = plt.subplots()
ax.set(title="Anscombe's trio and anti-trio", xlabel="x", ylabel="Y")
[plt.scatter(x, y, label="y{}".format(i + 1)) for i, y in enumerate(Y.T)]
ax.legend()
plt.show()


###############################################################################
# Classical statistics
# --------------------

# Pearson test
r_class = [pearsonr(y, x) for y in Y.T]
print("Pearson tests:")
print_results(r_class, vlabels, "r")

# Spearman test
rho_class = [spearmanr(y, x) for y in Y.T]
print("Spearman tests:")
print_results(rho_class, vlabels, "rho")

# Correction for multiple tests, using Bonferroni's method: multiply p-values
# by the number of tests.
n_tests = Y.shape[1]

p_corrected = np.array(r_class)[:, 1] * n_tests
print("\nPearson tests, after Bonferroni correction:")
print_pvals(p_corrected, vlabels)

p_corrected = np.array(rho_class)[:, 1] * n_tests
print("Spearman tests, after Bonferroni correction:")
print_pvals(p_corrected, vlabels)

# Bonferroni correction is a crucial step to adjust p-values of each variable
# for multiple tests, avoiding type-I errors (ie, spurious correlations).
# However, it makes the hypothesis that tests are independent, that is not the
# case here.


###############################################################################
# Permutation statistics
# ----------------------

n_perm = 10000
np.random.seed(17)

# Permutated Pearson test, corrected by Rmax
r_perm, p_perm = permutation_corr(Y, x=x, n=n_perm, side="two")
print("\nPermutated Pearson tests, with {} permutations:".format(n_perm))
print_results(np.c_[r_perm, p_perm], vlabels, "r")

# Permutated Spearman test, corrected by Rmax
rho_perm, p_perm = permutation_corr(
    Y,
    x=x,
    n=n_perm,
    corr="spearman",
    side="two",
)
print("Permutated Spearman tests, with {} permutations:".format(n_perm))
print_results(np.c_[rho_perm, p_perm], vlabels, "rho")

# The Rmax method of the permutation correlation tests is used for adjusting
# the p-values in a way that controls the family-wise error rate.
# It is more powerful than Bonferroni correction when different variables are
# correlated: it avoids type-I and type-II errors.


###############################################################################
# Reference
# ---------
# [1] Anscombe, "Graphs in Statistical Analysis", Am Stat, 1973.
