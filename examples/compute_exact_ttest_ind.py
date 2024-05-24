"""
===============================================================================
Compute exact Student's t-test for independent samples
===============================================================================

Compute classical, permutated and exact right-sided Student's t-tests, for
independent / unpaired samples.
"""
# Author: Quentin Barthélemy

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import t, ttest_ind, mannwhitneyu, shapiro, levene

from pypermut.misc import pvals_to_stars
from pypermut.stats import permutation_ttest_ind, permutation_mannwhitneyu


###############################################################################
# Univariate data
# ---------------

# Artificial small samples
s1 = np.array([7.5, 7.9, 8.3, 8.5, 8.7, 9.2, 9.6, 10.2])
s2 = np.array([5.9, 6.5, 6.9, 7.4, 7.8, 7.9, 8.1, 8.5, 8.7, 9.1, 9.9])

# Plot samples
fig, ax = plt.subplots()
ax.set(title="Boxplot of samples", xlabel="Samples", ylabel="Values")
box = ax.boxplot(np.array([s1, s2], dtype=object), medianprops={"color": "k"},
                 showmeans=True, meanline=True, meanprops={"color": "r"})
ax.plot([1.1, 1.9], [m.get_ydata()[0] for m in box.get("means")], c="r")
ax.scatter(np.ones(len(s1)), s1)
ax.scatter(2*np.ones(len(s2)), s2)
plt.show()


###############################################################################
# Student's t-tests for independent samples
# -----------------------------------------

# The null hypothesis is that these two samples s1 and s2 are drawn from the
# same Gaussian distribution.

# Classical test
t_class, p_class = ttest_ind(s1, s2, alternative="greater")
print("Classical t-test:\n t={:.2f}, p={:.3e} ({})"
      .format(t_class, p_class, pvals_to_stars(p_class)))

# Permutated test
np.random.seed(17)
n_perm = 10000
t_perm, p_perm = permutation_ttest_ind(s1, s2, n=n_perm, side="one")
print("Permutation t-test:\n t={:.2f}, p={:.3e} ({})"
      .format(t_perm[0], p_perm[0], pvals_to_stars(p_perm[0])))

# Exact test
t_exact, p_exact, t_dist = permutation_ttest_ind(
    s1,
    s2,
    n="all",
    side="one",
    return_dist=True,
)
print("Exact t-test:\n t={:.2f}, p={:.3e} ({})\n"
      .format(t_exact[0], p_exact[0], pvals_to_stars(p_exact[0])))

# Plot the difference between approximate and exact right-sided p-values
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(121, title="Approximate p-value", xlabel="t")
ax1.axvline(x=t_exact, c="r", label="Real statistic t")
df = len(s1) + len(s2) - 2
x = np.linspace(t.ppf(0.001, df), t.ppf(0.999, df), 100)
ax1.plot(x, t.pdf(x, df), label="Approximate t-distribution")
ax1.fill_between(x, t.pdf(x, df), where=(x>=t_exact), facecolor="r",
                 label="Approximate p-value")
plt.legend(loc="upper left")
ax2 = fig.add_subplot(122, title="Exact p-value", xlabel="t", sharey=ax1)
ax2.axvline(x=t_exact, c="r", label="Real statistic t")
y, x, _ = ax2.hist(t_dist, 50, density=True, histtype="step",
                   label="Exact t-distribution")
ax2.fill_between(x[1:], y, where=(x[1:]>=t_exact), facecolor="r", step="pre",
                 label="Exact p-value")
ax2.set_xlim(ax1.get_xlim())
plt.legend(loc="upper left")
plt.show()

# Classical t-test detects a trend in the difference between these samples,
# but is not able to reject the null hypothesis (with alpha = 0.05).
# Permutation and exact tests reject the null hypothesis, the later giving the
# exact p-value. Amazing!


###############################################################################
# Assumptions
# -----------

# But, wait! Is Student's t-test valid for such data?
# We have not checked the assumptions related to this parametric test.

alpha = 0.05

# Check homoscedasticity (homogeneity of variances) assumption, with a Levene's
# test
stat, p = levene(s1, s2)
print("Levene test:\n Equal var={}, p={:.3f} ({})"
      .format(p < alpha, p, pvals_to_stars(p)))

# Ok, but we could use a Welch's t-test, which does not assume equal variance.

# Check normality / Gaussianity assumption, with Shapiro-Wilk tests
stat, p = shapiro(s1)
print("Shapiro-Wilk test, for first sample:\n Normality={}, p={:.3f} ({})"
      .format(p < alpha, p, pvals_to_stars(p)))
stat, p = shapiro(s2)
print("Shapiro-Wilk test, for second sample:\n Normality={}, p={:.3f} ({})\n"
      .format(p < alpha, p, pvals_to_stars(p)))

# Consequently, neither Student's t-test nor Welch's t-test can be applied on
# these non-Gaussian data. We must use the non-parametric version.


###############################################################################
# Mann-Whitney U tests for unpaired samples
# -----------------------------------------

# The null hypothesis is that these two samples s1 and s2 are drawn from the
# same distribution, without assumption on its shape.

# Classical test
U_class, p_class = mannwhitneyu(s1, s2, alternative="less")
print("Classical Mann-Whitney test:\n U={:.1f}, p={:.3e} ({})"
      .format(U_class, p_class, pvals_to_stars(p_class)))

# Permutated test
U_perm, p_perm = permutation_mannwhitneyu(s1, s2, n=n_perm)
print("Permutation Mann-Whitney test:\n U={:.1f}, p={:.3e} ({})"
      .format(U_perm[0], p_perm[0], pvals_to_stars(p_perm[0])))

# Exact test
U_exact, p_exact = permutation_mannwhitneyu(s1, s2, n="all")
print("Exact Mann-Whitney test:\n U={:.1f}, p={:.3e} ({})"
      .format(U_exact[0], p_exact[0], pvals_to_stars(p_exact[0])))

# Finally, we must conclude that we can't reject the null hypothesis.


###############################################################################
