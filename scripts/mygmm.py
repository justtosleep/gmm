import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import logsumexp



def _n_parameters(means, covariance_type="full"):
    """Return the number of free parameters in the model."""
    n_components, n_features = means.shape
    if covariance_type == "full":
        cov_params = n_components * n_features * (n_features + 1) / 2.0
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) / 2.0
    elif covariance_type == "spherical":
        cov_params = n_components
    mean_params = n_features * n_components
    return int(cov_params + mean_params + n_components - 1)

def bic(X, means, precisions_chol, covariance_type="full", resp=None):
    """Bayesian information criterion for the current model on the input X.

    You can refer to this :ref:`mathematical section <aic_bic>` for more
    details regarding the formulation of the BIC used.

    Parameters
    ----------
    X : array of shape (n_samples, n_dimensions)
        The input samples.

    Returns
    -------
    bic : float
        The lower the better.
    """
    score = cal_score(X, means, precisions_chol, covariance_type, resp)
    bic = -2 * score * X.shape[0] + _n_parameters(means, covariance_type) * np.log(X.shape[0])
    return bic

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances

def _estimate_gmm_parameters(X, resp, reg_covar=1e-6):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
    return nk, means, covariances 

def m_setp(X, log_resp, covariance_type="full"):
    weights, means, covariances = _estimate_gmm_parameters(X, np.exp(log_resp), 1e-6)
    weights /= weights.sum()
    return weights, means, covariances

def cal_diff(means1, covariances1, means2, covariances2):
    # l1_means_dist = np.linalg.norm(means1-means2, ord=1, axis=1).sum()
    # percent_l1_means_diff1 = l1_means_dist/np.linalg.norm(means, ord=1, axis=1).sum()
    l1_means_dist = abs((means1-means2)).sum()
    percent_l1_means_diff1 = l1_means_dist/abs(means1).sum()*100
    # print(f"l1_means_dist({l1_means_dist:.4f})/means_dist({means1.sum():.4f})*100%={percent_l1_means_diff1:.4f}")
    # l1_covariances_dist = np.linalg.norm(covariances1-covariances2, ord=1, axis=(1,2)).sum()
    # percent_l1_covariances_diff1 = l1_covariances_dist/np.linalg.norm(covariances, ord=1, axis=(1,2)).sum()
    l1_covariances_dist = abs((covariances1-covariances2)).sum()
    percent_l1_covariances_diff1 = l1_covariances_dist/abs(covariances1).sum()*100
    # print(f"l1_covariances_dist({l1_covariances_dist:.4f})/covariances_dist({abs(covariances1).sum():.4f})*100%={percent_l1_covariances_diff1:.4f}")
    # print("covariances1\n", covariances1[0])
    # print("covariances2\n", covariances2[0])
    # print("covariance1-2\n", (covariances1-covariances2)[0])
    return percent_l1_means_diff1, percent_l1_covariances_diff1

def delete_small_cluster(means, covariances, idxs):
    means = np.delete(means, idxs, axis=0)
    covariances = np.delete(covariances, idxs, axis=0)
    return means, covariances