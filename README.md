# P*y*&#43004;*ermut*

[![Code PythonVersion](https://img.shields.io/badge/python-3.8+-blue)](https://img.shields.io/badge/python-3.8+-blue)
[![License](https://img.shields.io/badge/licence-BSD--3--Clause-green)](https://img.shields.io/badge/license-BSD--3--Clause-green)
[![Country](https://img.shields.io/badge/made%20in-France-blue)](https://img.shields.io/badge/made%20in-France-blue)

PyPermut is a Python package implementing permutation tests, for statistics and
machine learning.

PyPermut is distributed under the open source 3-clause BSD license.

## Description

In a nutshell, this package extends some parametric and non-parametric
[statistical tests](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests)
into permutation tests for multivariate samples, allowing the computation of
exact p-values and the correction for multiple tests.

Futhermore, this package offers several functions for machine learning, like a
generic permutation test for any metric based on the predictions of a model
(for eg, AUROC, AUPR, negative log-likelihood, etc).

#### Permutations for statistical tests

This package implements several
[permutation tests](https://en.wikipedia.org/wiki/Permutation_tests)
(also called randomization tests).

Correlation tests:
- Pearson product-moment correlation r test,
- Spearman rank-order correlation &#961; test.

Ordinal and numerical
[location tests](https://en.wikipedia.org/wiki/Location_test):
- Student's t-test for related samples,
- Student's t-test for independent samples
(and its adaptation into Welch's unequal variances t-test),
- Wilcoxon (signed-rank) T test,
- Mann-Whitney U test,
- one-way ANOVA for independent samples,
- Kruskal-Wallis H test,
- Friedman &#967;&#178; test.

In permutation tests, the null distribution is sampled by random permutations
of measurements. Making no hypothesis on the statistic distribution, the
p-value is defined as the proportion of the null distribution with test
statistic greater (or, for certain test statistic, lower) than or equal to the
test statistic of the observed data.
When all permutations are evaluated, these permutation tests give
[exact tests](https://en.wikipedia.org/wiki/Exact_test).

![](/doc/fig_approx_vs_exact.png)

In case of several variables, they naturally perform a correction for
[multiple tests](https://en.wikipedia.org/wiki/Multiple_comparisons_problem):
- rmax permutation test from Pearson r correlations,
- &#961;max permutation test from Spearman &#961; correlations,
- tmax permutation test from Student's t-tests for related samples,
- tmax permutation test from Student's t-tests for independent samples,
- Tmin permutation test from Wilcoxon T tests,
- Umin permutation test from Mann-Whitney U tests,
- Fmax permutation test from one-way ANOVAs for independent samples,
- Hmax permutation test from Kruskal-Wallis H tests,
- &#967;max permutation test from Friedman &#967;&#178; tests.

In the context of correlated variables, they efficiently correct the
[family-wise error rate](https://en.wikipedia.org/wiki/Family-wise_error_rate),
being an alternative to the well-known
[Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction)
which would lead to an undesirable lack of statistical power.
These multiple variables can be useful to compute pairwise post-hoc tests.

#### Decision tree for statistical tests

![](/doc/fig_tree_statistical_tests.svg)

#### How to use PyPermut

See `examples`.

#### References

Westfall and Young, *Resampling-based multiple testing: examples and methods*
*for p-value adjustment*, Wiley-Blackwell, 1993.

Pesarin, *Multivariate Permutation Tests – With Applications in Biostatistics*,
Wiley-Blackwell, 2001.

Nichols and Holmes, "Nonparametric permutation tests for functional
neuroimaging: A primer with examples", *Human Brain Mapping*, vol 15, pp 1-25,
2001.

Edgington and Onghena, *Randomization Tests, Fourth Edition*, Chapman and
Hall/CRC, 2007.

## Documentation

TODO: put documentation on RTD.

## Installation

#### From sources

To install PyPermut as a standard module:
```shell 
pip install path/to/pypermut
```

To install PyPermut in editable / development mode, in the folder:
```shell
pip install poetry
poetry install
```

## Testing

Use `pytest`.

## Authors

Quentin Barthélemy, an anonymous researcher and Louis Mayaud,
under the wisdom of [Marco Congedo](https://github.com/Marco-Congedo).
