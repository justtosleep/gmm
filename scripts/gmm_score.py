# GaussianMixture Clustering
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
sys.path.append("..")
from scripts.tools import create_directory
from sklearn.preprocessing import MinMaxScaler
# from my_sklearn.sklearn.mixture import GaussianMixture
from sklearn.mixture import GaussianMixture
from scipy import linalg
from scipy.special import logsumexp

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
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

def _compute_precision_cholesky(covariances, covariance_type):
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol

def _estimate_gmm_parameters(X, resp, reg_covar, covariance_type):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances 

def cal_score(X, means, precisions_chol, covariance_type, resp=None):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision
    # matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    if covariance_type == "full":
        n_components, _, _ = precisions_chol.shape
        #caulculate log_det
        log_det = np.sum(
            np.log(precisions_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )
        #calculate log_prob
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)
    elif covariance_type == "tied":
        #calculate log_det
        log_det = np.sum(np.log(np.diag(precisions_chol)))
        #calculate log_prob
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)
    elif covariance_type == "diag":
        #calculate log_det
        log_det = np.sum(np.log(precisions_chol), axis=1)
        #calculate log_prob
        precisions = precisions_chol**2
        log_prob = (
            np.sum((means**2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X**2, precisions.T)
        )
    else:
        #calculate log_det
        log_det = n_features * (np.log(precisions_chol))
        #calculate log_prob
        precisions = precisions_chol**2
        log_prob = (
            np.sum(means**2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            # + np.outer(row_norms(X, squared=True), precisions)
        )
    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    log_proba =  -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    #calculate weights
    if resp is None:
        weights = np.ones((n_samples, n_components))
        weights /= weights.sum()
    else:
        weights = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

    weighted_log_prob = log_proba + np.log(weights)

    return logsumexp(weighted_log_prob, axis=1).mean(), log_prob, weighted_log_prob, np.log(weights)

def m_setp(X, log_resp, covariance_type):
    weights, means, covariances = _estimate_gmm_parameters(X, np.exp(log_resp), 1e-6, covariance_type)
    weights /= weights.sum()
    precisions_cholesky = _compute_precision_cholesky(covariances, covariance_type)
    return weights, means, covariances, precisions_cholesky

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

# read command line arguments
parser = argparse.ArgumentParser(description="GaussianMixture Clustering")

# add positional arguments

# add optional arguments
parser.add_argument("--dataset", type=str, help="name of dataset")
parser.add_argument("--data_path", type=str, help="path of dataset")
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
data_path = args.data_path
method = args.method
print("domain: ", domain)
print("data path: ", data_path)
print("method: ", method)

# 1. Read data
print("Start reading data")
name = domain+"_"+method if method != "" else domain
input_path = "../dataset/" + data_path
data = pd.read_csv(input_path, skipinitialspace=True)
print("data shape", data.shape)
if method == "famd":
    minmax_scaler = MinMaxScaler()
    data = minmax_scaler.fit_transform(data)
else:
    data = data

# 2. Set parameters
data1_path = "../dataset/" + data_path.split('.')[0]+"_sample1.data"
data1 = pd.read_csv(data1_path, skipinitialspace=True)
# method1 = "sample1"
data2_path = "../dataset/" + data_path.split('.')[0]+"_sample2.data"
# method2 = "sample2"
data2 = pd.read_csv(data2_path, skipinitialspace=True)
sample_num = data1.shape[0]

# n_components = list(range(2, 11))+[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, \
                                            # 130, 140, 150, 160, 170, 180, 190, 200]
                                            # , 300, \
                                            # 400, 500, 600, 700, 800, 900, 1000]
n_components = [200,2,3,5,10,50,100]
# max_iters = [100,300,500,1000]  
max_iters = [100]  
# covariance_types = ['full', 'tied', 'diag', 'spherical']
covariance_types = ['full']
# output_dir = "../result/"+name+"/prediction/"
# prediction_dir = output_dir+"y_pred/"
# probability_dir = output_dir+"y_prob/"
# if not os.path.exists(prediction_dir):
#     create_directory(prediction_dir)
# if not os.path.exists(probability_dir):
#     create_directory(probability_dir)

output_dir = "../result/"+name+"/iter/"
data_dir = "../result/"+name+"/data/"
# output_dir = "../result/score/"
if not os.path.exists(output_dir):
    create_directory(output_dir)
if not os.path.exists(data_dir):
    create_directory(data_dir)
iters_path = output_dir+f"iter.csv"
with open(iters_path, 'w') as f:
    f.write("max_iter,actual_iter,coverage,n_components,l1_means_diff1,l1_means_diff2,l1_covariances_diff1,l1_covariances_diff2\n")

# 3. Apply GaussianMixture
begin_time = time.time()
print("Start applying GaussianMixture")
for max_iter in max_iters:
    for covariance_type in covariance_types:
        for n_component in n_components:
            # print(f"Trying n_component={n_component}, covariance_type={covariance_type}, n_init={n_init}")
            start_time = time.time()
            #GMM
            #If reg_covar is not set, the vermont-diag will fail
            model = GaussianMixture(n_components=n_component, \
                                    n_init=1, \
                                    covariance_type=covariance_type, \
                                    max_iter=max_iter, \
                                    random_state=42
                                    )
            model.fit(data1)
            elapsed_time = time.time() - start_time
            means = model.means_
            covariances = model.covariances_
            precisions_chol = model.precisions_cholesky_
            coverage = model.converged_
            actual_iter = model.n_iter_
            score = model.score(data1)
            np.save(data_dir+f"means_{n_component}.npy", means)
            np.save(data_dir+f"covariances_{n_component}.npy", covariances)
            np.save(data_dir+f"precisions_chol_{n_component}.npy", precisions_chol)
            
            resp = model.predict_proba(data1)
            resp += 10 * np.finfo(resp.dtype).eps
            resp /= resp.sum()
            log_resp = np.log(resp)
            resp1, means1, covariances1, precisions_chol1 = m_setp(data1, log_resp, covariance_type)
            l1_means_diff1, l1_covariances_diff1 = cal_diff(means, covariances, means1, covariances1)
            np.save(data_dir+f"class_proportions1_log_resp_{n_component}.npy", log_resp)
            np.save(data_dir+f"class_proportions1_resp_{n_component}.npy", resp1)
            np.save(data_dir+f"class_proportions1_means_{n_component}.npy", means1)
            np.save(data_dir+f"class_proportions1_covariances_{n_component}.npy", covariances1)
            np.save(data_dir+f"class_proportions1_precisions_chol_{n_component}.npy", precisions_chol1)

            resp = model.predict_proba(data2)
            resp += 10 * np.finfo(resp.dtype).eps
            resp /= resp.sum()
            log_resp = np.log(resp)
            resp2, means2, covariances2, precisions_chol2 = m_setp(data2, log_resp, covariance_type)
            l1_means_diff2, l1_covariances_diff2 = cal_diff(means, covariances, means2, covariances2)
            np.save(data_dir+f"class_proportions2_log_resp_{n_component}.npy", log_resp)
            np.save(data_dir+f"class_proportions2_resp_{n_component}.npy", resp2)
            np.save(data_dir+f"class_proportions2_means_{n_component}.npy", means2)
            np.save(data_dir+f"class_proportions2_covariances_{n_component}.npy", covariances2)
            np.save(data_dir+f"class_proportions2_precisions_chol_{n_component}.npy", precisions_chol2)

            labels1 = model.predict(data1)
            unique_labels1, label_counts1 = np.unique(labels1, return_counts=True)
            total_samples1 = labels1.shape[0] 
            label_proportions1 = label_counts1 / total_samples1
            plt.clf()
            plt.bar(unique_labels1, label_proportions1)
            plt.xticks(unique_labels1)
            plt.xlabel("Class")
            plt.ylabel("Proportion")
            graph_title = f"converaged in {model.n_iter_} iterations" if model.converged_ else f"not converaged in {model.n_iter_} iterations"
            plt.title(graph_title)
            graph_dir = "../result/"+name+"/graph/"
            if not os.path.exists(graph_dir):
                create_directory(graph_dir)
            plt.savefig("../result/"+name+f"/graph/class_proportions1_{n_component}.png")
            np.save(data_dir+f"class_proportions1_unique_labels_{n_component}.npy", unique_labels1)
            np.save(data_dir+f"class_proportions1_label_counts_{n_component}.npy", label_counts1)
            np.save(data_dir+f"class_proportions1_label_proportions_{n_component}.npy", label_proportions1)

            labels2 = model.predict(data2)
            unique_labels2, label_counts2 = np.unique(labels2, return_counts=True)
            total_samples2 = labels2.shape[0]
            label_proportions2 = label_counts2 / total_samples2
            plt.clf()
            plt.bar(unique_labels2, label_proportions2)
            plt.xticks(unique_labels2)
            plt.xlabel("Class")
            plt.ylabel("Proportion")
            graph_title = f"converaged in {model.n_iter_} iterations" if model.converged_ else f"not converaged in {model.n_iter_} iterations"
            plt.title(graph_title)
            plt.savefig("../result/"+name+f"/graph/class_proportions2_{n_component}.png")
            np.save(data_dir+f"class_proportions2_unique_labels_{n_component}.npy", unique_labels2)
            np.save(data_dir+f"class_proportions2_label_counts_{n_component}.npy", label_counts2)
            np.save(data_dir+f"class_proportions2_label_proportions_{n_component}.npy", label_proportions2)

            print(f"Trying n_component={n_component}, score={score}")
            with open(iters_path, 'a') as f:
                f.write(f"{max_iter},{actual_iter},{coverage},{n_component},{l1_means_diff1:.4f},{l1_means_diff2:.4f},{l1_covariances_diff1:.4f},{l1_covariances_diff2:.4f}\n")

    print(f"{covariance_type} finished, time elapsed: {time.time()-begin_time:.2f} seconds")
end_time = time.time() - begin_time
print(f"Total time elapsed: {end_time:.2f} seconds")

