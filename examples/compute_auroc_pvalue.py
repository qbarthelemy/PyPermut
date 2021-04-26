"""
===============================================================================
Compute the p-value associated to AUROC
===============================================================================

Compute the p-value associated to the area under the curve (AUC) of the 
receiver operating characteristic (ROC), for a binary (two-class)
classification problem.

Warning: this example requires scikit-learn dependency (0.22 at least).
"""
# Authors: Quentin Barthélemy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from pypermut.ml import permutation_metric, standard_error_auroc
from pypermut.misc import pvals_to_stars
from matplotlib import pyplot as plt


###############################################################################
# Artificial data and classifier
# ------------------------------

X, y = make_classification(n_classes=2, n_samples=100, n_features=100,
                           n_informative=10, n_redundant=10,
                           n_clusters_per_class=5, random_state=1)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5,
                                                random_state=2)
# Logistic regression: a very good old-fashioned classifier
clf = LogisticRegression().fit(Xtrain, ytrain)
ytest_probas = clf.predict_proba(Xtest)[:, 1]


###############################################################################
# Compute ROC curve and AUC
# -------------------------

# In this example, we focus on the AUROC metric. However, this analysis can be
# applied to any metric measuring the model performance. It can be any function
# from sklearn.metrics, like roc_auc_score, log_loss, etc.

# ROC curve
fpr, tpr, _ = roc_curve(ytest, ytest_probas)

fig, ax = plt.subplots()
ax.set(title="ROC curve", xlabel='FPR', ylabel='TPR')
ax.plot(fpr, tpr, 'C1', label='Logistic regression')
ax.plot([0, 1], [0, 1], 'b--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()

# AUC
auroc = roc_auc_score(ytest, ytest_probas)
print('AUROC = {:.3f}\n'.format(auroc))


###############################################################################
# Compute standard error and p-value of AUC
# -----------------------------------------

n_perm = 5000
np.random.seed(17)

# standard error of AUC
auroc_se = standard_error_auroc(ytest, ytest_probas, auroc)

# p-value of AUC, by permutations
auroc_, pval = permutation_metric(ytest, ytest_probas, roc_auc_score,
                                  side='right', n=n_perm)
print('AUROC = {:.3f} +/- {:.3f}, p={:.2e} ({})'
      .format(auroc_, auroc_se, pval, pvals_to_stars(pval)))


###############################################################################
# Reference
# ---------
# [1] Hanley & McNeil, "The meaning and use of the area under a receiver
# operating characteristic (ROC) curve", Radiology, 1982.
