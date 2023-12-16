import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import logsumexp
import torch


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = torch.empty((n_components, n_features, n_features), dtype=X.dtype)
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = torch.mm(resp[:, k] * diff.t(), diff) / nk[k]
        covariances[k].view(-1)[::n_features + 1] += reg_covar
    return covariances

def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance

def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)

def _estimate_gmm_parameters_torch(X, resp, reg_covar, covariance_type):
    nk = resp.sum(dim=0) + 10 * torch.finfo(resp.dtype).eps
    means = (resp.t() @ X) / nk.view(-1, 1)
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        # "tied": _estimate_gaussian_covariances_tied,
        # "diag": _estimate_gaussian_covariances_diag,
        # "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances 


def check_covs_torch(covariances):
    # check if all value in covariances is between -1 and 1
    return torch.all((covariances >= -1) & (covariances <= 1))

def m_setp_torch(X, log_resp, covariance_type="full"):
    weights, means, covariances = _estimate_gmm_parameters_torch(X, np.exp(log_resp), 1e-6, covariance_type)
    weights /= weights.sum()
    return weights, means, covariances

def cal_diff_torch(means1, covariances1, means2, covariances2):
    # l1_means_dist = np.linalg.norm(means1-means2, ord=1, axis=1).sum()
    # percent_l1_means_diff1 = l1_means_dist/np.linalg.norm(means, ord=1, axis=1).sum()
    l1_means_dist = torch.abs((means1-means2)).sum()
    percent_l1_means_diff1 = l1_means_dist/torch.abs(means1).sum()*100
    # print(f"l1_means_dist({l1_means_dist:.4f})/means_dist({means1.sum():.4f})*100%={percent_l1_means_diff1:.4f}")
    # l1_covariances_dist = np.linalg.norm(covariances1-covariances2, ord=1, axis=(1,2)).sum()
    # percent_l1_covariances_diff1 = l1_covariances_dist/np.linalg.norm(covariances, ord=1, axis=(1,2)).sum()
    l1_covariances_dist = torch.abs((covariances1-covariances2)).sum()
    percent_l1_covariances_diff1 = l1_covariances_dist/torch.abs(covariances1).sum()*100
    # print(f"l1_covariances_dist({l1_covariances_dist:.4f})/covariances_dist({torch.abs(covariances1).sum():.4f})*100%={percent_l1_covariances_diff1:.4f}")
    # print("covariances1\n", covariances1[0])
    # print("covariances2\n", covariances2[0])
    # print("covariance1-2\n", (covariances1-covariances2)[0])
    return percent_l1_means_diff1, percent_l1_covariances_diff1

def cal_abs_diff_torch(means1, covariances1, means2, covariances2):
    n, d, _ = covariances1.shape
    n2, d2, _ = covariances2.shape
    if n != n2 or d != d2:
        print("n1, n2, d1, d2", n, n2, d, d2)
        raise ValueError("covariances1 and covariances2 should have same shape")
    # avg_means = torch.abs(means)
    l1_means_dist = torch.abs(means1-means2).sum() / (n*d)
    l1_covariances_dist = torch.abs(covariances1-covariances2).sum() / (n*d*d)
    return l1_means_dist, l1_covariances_dist
