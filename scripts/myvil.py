import matplotlib.pyplot as plt
import numpy as np
import os
from mygmm import cal_diff, m_setp, bic
from mygmm_torch import cal_diff_torch, m_setp_torch, check_covs_torch, cal_abs_diff_torch
import torch

means_prefix = "means_"
covariances_prefix = "covariances_"
precisions_chol_prefix = "precisions_chol_"

log_resp1_prefix = "class_proportions1_log_resp_"
label_counts1_prefix = "class_proportions1_label_counts_"
unique_labels1_prefix = "class_proportions1_unique_labels_"

log_resp2_prefix = "class_proportions2_log_resp_"
label_counts2_prefix = "class_proportions2_label_counts_"
unique_labels2_prefix = "class_proportions2_unique_labels_"

def vil_cluster_proportion(unique_labels, label_counts, n_component, graph_dir, sample="sample0", coverage_iter=0, coverage=True): 
    label_proportions = label_counts / label_counts.sum()
    plt.clf()
    plt.bar(unique_labels, label_proportions)
    # plt.xticks(unique_labels)
    plt.xlabel("Class")
    plt.ylabel("Proportion")
    # graph_title = f"converaged in {coverage_iter} iterations" if coverage else f"not converaged in {coverage_iter} iterations"
    # plt.title(graph_title)
    plt.savefig(graph_dir+f"class_proportions{sample[-1]}_{n_component}.png")

def _find_median(data):
    n = len(data)
    if n < 1:
        return 0
    if n % 2 == 0:
        mid1 = data[n // 2]
        mid2 = data[n // 2 - 1]
        median = (mid1 + mid2) / 2
    else:
        median = data[n // 2]
    return median

def vil_diff(n, m, i, x, y, n_component, ds, col, sample, plot_type="line"):
    title = f'{ds} {col}{sample[-1]}' if "diff" in col else f'{col} l1'
    xlabel = 'Minimal cluster Size' if "diff" in col else ''
    mean_cluster = np.mean(x)
    median_cluster = _find_median(x)
    ax = plt.subplot(n, m, i)
    if plot_type == "line":
        ax.plot(x, y, color='blue', label=f'avg_cluster:{mean_cluster:.2f}\nmedian_cluster:{median_cluster:.2f}')
    elif plot_type == "scatter":
        ax.scatter(x, y, color='blue', marker='o', label=f'avg_cluster:{mean_cluster:.2f}\nmedian_cluster:{median_cluster:.2f}')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    # fontsize = 14
    # plt.sca(ax)

def vil_cluster_size(label_counts1, label_counts2, n_component, graph_dir, name=""):
    plt.clf()
    plt.subplot(1, 2, 1)
    x_values = list(range(max(label_counts1)))
    y_values = [sum(1 for size in label_counts1 if size > x) for x in x_values]
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title(f'{name} Sample1 Cluster Size')
    plt.xlabel('Cluster Size')
    plt.ylabel('Number of Clusters (Size > x)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x_values = list(range(max(label_counts2)))
    y_values = [sum(1 for size in label_counts2 if size > x) for x in x_values]
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title(f'{name} Sample2 Cluster Size')
    plt.xlabel('Cluster Size')
    plt.ylabel('Number of Clusters (Size > x)')
    plt.grid(True)
    output_path = graph_dir + f"vil_cluster_size_{n_component}.png"
    plt.savefig(output_path)

def _erase_component(filename):
    number = filename.split('.')[0].split('_')[-1]
    return filename.replace(number, '').split('.')[0]

def _file_prefix(folder_path):
    flag = [0]*4
    for file in os.listdir(folder_path):
        if file.startswith("covariances") and not flag[0]:
            covariances_prefix = _erase_component(file)
            flag[0] = 1
        elif "1_log_resp" in file and not flag[1]:
            log_resp1_prefix = _erase_component(file)
            flag[1] = 1
        elif "2_log_resp" in file and not flag[2]:
            log_resp2_prefix = _erase_component(file)
            flag[2] = 1
        elif file.startswith("precisions_chol") and not flag[3]:
            precisions_chol_prefix = _erase_component(file)
            flag[3] = 1
        if sum(flag) == 4:
            break
    return covariances_prefix, log_resp1_prefix, log_resp2_prefix, precisions_chol_prefix


def means_list(folder_path):
    means_files = [x for x in os.listdir(folder_path) if x.startswith(means_prefix)]
    x_list = []
    for i, m_path in enumerate(means_files):
        n_component = m_path.split('_')[-1].split('.')[0]
        x_list.append(int(n_component))
    idxs = np.argsort(x_list)
    x_list = np.array(x_list)[idxs]
    means_files = np.array(means_files)[idxs]
    return means_files, x_list

def resp2_list(folder_path):
    resp_files = [x for x in os.listdir(folder_path) if x.startswith(log_resp2_prefix)]
    x_list = []
    for i, m_path in enumerate(resp_files):
        n_component = m_path.split('_')[-1].split('.')[0]
        x_list.append(int(n_component))
    idxs = np.argsort(x_list)
    x_list = np.array(x_list)[idxs]
    resp_files = np.array(resp_files)[idxs]
    return resp_files, x_list

def diff_seed_noise(org_data_dir, folder_path, data1, data2, domain, n_components=None):
    n = data2.shape[0]
    threshold = 0.0001
    _, x_list = resp2_list(folder_path)
    if n_components is not None:
        x_list = n_components
    print("Reading x_list...", x_list)
    # print("Reading resp2 files...", resp2_files)

    means_diff1_list = []
    covariances_diff1_list = []
    means_diff2_list = []
    covariances_diff2_list = []
    original_data_path = folder_path
    for n_component in x_list:
        try:
            means_path = f"{means_prefix}{n_component}.npy"
            covs_path = f"{covariances_prefix}{n_component}.npy"
            log_resp1_path = f"{log_resp1_prefix}{n_component}.npy"
            means = np.load(org_data_dir + means_path)
            covs = np.load(org_data_dir + covs_path)
            log_resp1 = np.load(org_data_dir + log_resp1_path)
            
            log_resp2_path = f"{log_resp2_prefix}{n_component}.npy"
            label_counts2_path = f"{label_counts2_prefix}{n_component}.npy"
            unique_labels2_path = f"{unique_labels2_prefix}{n_component}.npy"
            log_resp2 = np.load(folder_path + log_resp2_path)
            label_counts2 = np.load(folder_path + label_counts2_path)
            unique_labels2 = np.load(folder_path + unique_labels2_path)
        except Exception as e:
            print(e)
            continue
        outlier_idxs = np.where(label_counts2 < threshold*n)[0]
        # print("outlier_idxs: ", outlier_idxs)
        # print("label_counts2: ", list(zip(unique_labels2, label_counts2)))
        # print("number of outlier point", list(zip(unique_labels2, label_counts2[outlier_idxs])))
        # print("len unique_labels2: ", len(unique_labels2))
        # print("unique_labels2: ", unique_labels2)
        unique_labels2 = np.delete(unique_labels2, outlier_idxs)
        # print("len unique_labels2: ", len(unique_labels2))
        # print("unique_labels2: ", unique_labels2)

        means = means[unique_labels2]
        covs = covs[unique_labels2]
        _, means1, covariances1 = m_setp(data1, log_resp1)
        means1 = means1[unique_labels2]
        covariances1 = covariances1[unique_labels2]
        l1_means_diff1, l1_covariances_diff1 = cal_diff(means, covs, means1, covariances1)
        means_diff1_list.append(l1_means_diff1)
        covariances_diff1_list.append(l1_covariances_diff1)

        _, means2, covariances2= m_setp(data2, log_resp2)
        means2 = means2[unique_labels2]
        covariances2 = covariances2[unique_labels2]
        l1_means_diff2, l1_covariances_diff2 = cal_diff(means, covs, means2, covariances2)
        means_diff2_list.append(l1_means_diff2)
        covariances_diff2_list.append(l1_covariances_diff2)
        # ####################################
        # ###Print the outlier points        ###
        # print("means2.shape: ", means2.shape)
        # print("l1_covariances_diff2: ", l1_covariances_diff2)
        # diffs = abs(covs-covariances2).sum((1,2))
        # cov_diffs = ["{:2e}".format(diff) for diff in diffs]
        # l1_diffs = diffs/abs(covs).sum()*100
        # l1_diffs = ["{:2e}".format(diff) for diff in l1_diffs]
        # print("abs(covs-covs2)", list(zip(unique_labels2, cov_diffs)))
        # print("abs(covs)", "{:2e}".format(abs(covs).sum()))
        # print("abs(covs-covs2)/abs(covs)", list(zip(unique_labels2, l1_diffs)))
        # ####################################
    
    return x_list, means_diff1_list, covariances_diff1_list, means_diff2_list, covariances_diff2_list

def diff_seed_noise_torch(org_data_dir, folder_path, data1, data2, domain, n_components=None):
    data1 = torch.from_numpy(data1.values).double()
    data2 = torch.from_numpy(data2.values).double()
    n, _ = data2.shape
    threshold = 0.0001
    _, x_list = resp2_list(folder_path)
    if n_components is not None:
        x_list = n_components
    print("target clusters...", x_list)
    # print("Reading resp2 files...", resp2_files)

    means_diff1_list = []
    covariances_diff1_list = []
    means_diff2_list = []
    covariances_diff2_list = []
    original_data_path = folder_path
    for n_component in x_list:
        try:
            means_path = f"{means_prefix}{n_component}.npy"
            covs_path = f"{covariances_prefix}{n_component}.npy"
            log_resp1_path = f"{log_resp1_prefix}{n_component}.npy"

            means = np.load(org_data_dir + means_path)
            means = torch.from_numpy(means)
            covs = np.load(org_data_dir + covs_path)
            covs = torch.from_numpy(covs)
            log_resp1 = np.load(org_data_dir + log_resp1_path)
            log_resp1 = torch.from_numpy(log_resp1)
            
            log_resp2_path = f"{log_resp2_prefix}{n_component}.npy"
            label_counts2_path = f"{label_counts2_prefix}{n_component}.npy"
            unique_labels2_path = f"{unique_labels2_prefix}{n_component}.npy"

            log_resp2 = np.load(folder_path + log_resp2_path)
            log_resp2 = torch.from_numpy(log_resp2)
            label_counts2 = np.load(folder_path + label_counts2_path)
            label_counts2 = torch.from_numpy(label_counts2)
            unique_labels2 = np.load(folder_path + unique_labels2_path)
            unique_labels2 = torch.from_numpy(unique_labels2)
        except Exception as e:
            print(e)
            continue
        # outlier_idxs = np.where(label_counts2 < threshold*n)[0]
        outlier_idxs = torch.where(label_counts2 < threshold*n)[0]
        print(f"{len(outlier_idxs)} outlier clusters: ", outlier_idxs)
        print("(clusters, number of point)", list(zip(unique_labels2, label_counts2)))
        print("(outlier clusters, number of point)", list(zip(unique_labels2[outlier_idxs], label_counts2[outlier_idxs])))

        print(f"before removing outliers, there is {len(unique_labels2)} clusters", unique_labels2)
        unique_labels2 = unique_labels2[torch.tensor([i for i in range(len(unique_labels2)) if i not in outlier_idxs])]
        print(f"after removing outliers, there is {len(unique_labels2)} clusters", unique_labels2)

        means = means[unique_labels2]
        covs = covs[unique_labels2]
        _, means1, covariances1 = m_setp_torch(data1, log_resp1)
        means1 = means1[unique_labels2]
        covariances1 = covariances1[unique_labels2]
        check_covs_torch(covariances1)
        # l1_means_diff1, l1_covariances_diff1 = cal_diff_torch(means, covs, means1, covariances1)
        l1_means_diff1, l1_covariances_diff1 = cal_abs_diff_torch(means, covs, means1, covariances1)
        means_diff1_list.append(l1_means_diff1)
        covariances_diff1_list.append(l1_covariances_diff1)

        _, means2, covariances2= m_setp_torch(data2, log_resp2)
        means2 = means2[unique_labels2]
        covariances2 = covariances2[unique_labels2]
        check_covs_torch(covariances2)
        # l1_means_diff2, l1_covariances_diff2 = cal_diff_torch(means, covs, means2, covariances2)
        l1_means_diff2, l1_covariances_diff2 = cal_abs_diff_torch(means, covs, means2, covariances2)
        means_diff2_list.append(l1_means_diff2)
        covariances_diff2_list.append(l1_covariances_diff2)
        ####################################
        #### Print the outlier points        ###
        n_clusters, d_features = means.shape
        # print("means", means)
        # print("abs(means)", "{:2e}".format(abs(means).sum()))
        print("avg(abs(means))", "{:2e}".format(abs(means).sum() / (n_clusters*d_features)))
        # print("abs(means1)", "{:2e}".format(abs(means1).sum()))
        print("avg(abs(means1))", "{:2e}".format(abs(means1).sum() / (n_clusters*d_features)))
        # print("abs(means2)", "{:2e}".format(abs(means2).sum()))
        print("avg(abs(means2))", "{:2e}".format(abs(means2).sum() / (n_clusters*d_features)))
        print("avg(abs(means-means1))", 
            "{:2e}".format(abs(means-means1).sum() / (n_clusters*d_features)))
        print("avg(abs(means-means2))",
            "{:2e}".format(abs(means-means2).sum() / (n_clusters*d_features)))
        # print("l1_means_abs1: ", l1_means_abs1)
        # print("l1_means_diff1: ", l1_means_diff1)
        # print("l1_means_abs2: ", l1_means_abs2)
        # print("l1_means_diff2: ", l1_means_diff2)
        # print("covs1 change over 0.1 paris", torch.sum(change_covs_pair1 > 0.1).item())
        # print("covs1 change over 0.01 paris", torch.sum(change_covs_pair1 > 0.01).item())
        # print("covs1 change over 0.001 paris", torch.sum(change_covs_pair1 > 0.001).item())
        # print("covs1 change over 0.0001 paris", torch.sum(change_covs_pair1 > 0.0001).item())

        # change_covs_pair2 = abs(covs-covariances2)
        # print("covs2 change over 0.1 paris", torch.sum(change_covs_pair2 > 0.1).item())
        # print("covs2 change over 0.01 paris", torch.sum(change_covs_pair2 > 0.01).item())
        # print("covs2 change over 0.001 paris", torch.sum(change_covs_pair2 > 0.001).item())
        # print("covs2 change over 0.0001 paris", torch.sum(change_covs_pair2 > 0.0001).item())
        # print("covs", covs)
        # print("covs1", covariances1)
        # print("covs2", covariances2)
        # print("abs(covs)", "{:2e}".format(abs(covs).sum()))
        print("avg(abs(covs))", "{:2e}".format(abs(covs).sum() / (n_clusters*d_features*d_features)))
        # print("abs(covariances1)", "{:2e}".format(abs(covariances1).sum()))
        print("avg(abs(covariances1))", "{:2e}".format(abs(covariances1).sum() / (n_clusters*d_features*d_features)))
        # print("abs(covariances2)", 
            # "{:2e}".format(abs(covariances2).sum()))
        print("avg(abs(covariances2))", 
            "{:2e}".format(abs(covariances2).sum() / (n_clusters*d_features*d_features)))
        print("avg(abs(covs-covariances1))",
            "{:2e}".format(abs(covs-covariances1).sum() / (n_clusters*d_features*d_features)))
        print("avg(abs(covs-covariances2))",
            "{:2e}".format(abs(covs-covariances2).sum() / (n_clusters*d_features*d_features)))
        # print("l1_covariances_abs1: ", l1_covariances_abs1)
        # print("l1_covariances_diff1: ", l1_covariances_diff1)
        # print("l1_covariances_abs2: ", l1_covariances_abs2)     
        # print("l1_covariances_diff2: ", l1_covariances_diff2)
        # diffs = abs(covs-covariances2).sum((1,2))
        # cov_diffs = ["{:2e}".format(diff) for diff in diffs]
        # l1_diffs = diffs/abs(covs).sum()*100
        # l1_diffs = ["{:2e}".format(diff) for diff in l1_diffs]
        # print("abs(covs-covs2)", list(zip(unique_labels2, cov_diffs)))
        # print("abs(covs-covs2)/abs(covs)", list(zip(unique_labels2, l1_diffs)))
        ####################################
    return x_list, means_diff1_list, covariances_diff1_list, means_diff2_list, covariances_diff2_list


def diff_seed(folder_path, data1, data2, domain):
    n = data2.shape[0]
    threshold = 0.0001
    means_files, x_list = means_list(folder_path)
    print("Reading x_list...", x_list)
    print("Reading means files...", means_files)

    means_diff1_list = []
    covariances_diff1_list = []
    means_diff2_list = []
    covariances_diff2_list = []
    covariances_prefix , log_resp1_prefix, log_resp2_prefix, _ = _file_prefix(folder_path)
    # print("Reading covariances files...", covariances_prefix)
    # print("Reading log_resp1 files...", log_resp1_prefix)
    # print("Reading log_resp2 files...", log_resp2_prefix)
    for i, m_path in enumerate(means_files):
        n_component = x_list[i]
        try:
            covs_path = f"{covariances_prefix}{n_component}.npy"
            log_resp1_path = f"{log_resp1_prefix}{n_component}.npy"
            log_resp2_path = f"{log_resp2_prefix}{n_component}.npy"
            covs = np.load(folder_path + covs_path)
            log_resp1 = np.load(folder_path + log_resp1_path)
            log_resp2 = np.load(folder_path + log_resp2_path)
            
            label_counts2 = np.load(folder_path+f"class_proportions2_label_counts_{n_component}.npy")
            unique_labels2 = np.load(folder_path+f"class_proportions2_unique_labels_{n_component}.npy")
        except:
            print(f"Cannot find n_component {n_component} in {domain}")
            continue
        means = np.load(folder_path + m_path)
        outlier_idxs = np.where(label_counts2 < threshold*n)[0]
        # print("outlier_idxs: ", outlier_idxs)
        # print("label_counts2: ", list(zip(unique_labels2, label_counts2)))
        # print("number of outlier point", list(zip(unique_labels2, label_counts2[outlier_idxs])))
        # print("len unique_labels2: ", len(unique_labels2))
        # print("unique_labels2: ", unique_labels2)
        unique_labels2 = np.delete(unique_labels2, outlier_idxs)
        # print("len unique_labels2: ", len(unique_labels2))
        # print("unique_labels2: ", unique_labels2)

        means = means[unique_labels2]
        covs = covs[unique_labels2]
        _, means1, covariances1 = m_setp(data1, log_resp1)
        means1 = means1[unique_labels2]
        covariances1 = covariances1[unique_labels2]
        l1_means_diff1, l1_covariances_diff1 = cal_diff(means, covs, means1, covariances1)
        means_diff1_list.append(l1_means_diff1)
        covariances_diff1_list.append(l1_covariances_diff1)

        _, means2, covariances2 = m_setp(data2, log_resp2)
        means2 = means2[unique_labels2]
        covariances2 = covariances2[unique_labels2]
        l1_means_diff2, l1_covariances_diff2 = cal_diff(means, covs, means2, covariances2)
        means_diff2_list.append(l1_means_diff2)
        covariances_diff2_list.append(l1_covariances_diff2)
        #####################################
        # Print the outlier points        ###
        # print("means2.shape: ", means2.shape)
        # print("l1_covariances_diff2: ", l1_covariances_diff2)
        # diffs = abs(covs-covariances2).sum((1,2))
        # cov_diffs = ["{:2e}".format(diff) for diff in diffs]
        # l1_diffs = diffs/abs(covs).sum()*100
        # l1_diffs = ["{:2e}".format(diff) for diff in l1_diffs]
        # print("abs(covs-covs2)", list(zip(unique_labels2, cov_diffs)))
        # print("abs(covs)", "{:2e}".format(abs(covs).sum()))
        # print("abs(covs-covs2)/abs(covs)", list(zip(unique_labels2, l1_diffs)))
        #####################################
    
    return x_list, means_diff1_list, covariances_diff1_list, means_diff2_list, covariances_diff2_list

def vil_elbow(folder_path, data1, domain, graph_dir):
    means_files, x_list = means_list(folder_path)
    print("Reading x_list...", x_list)
    print("Reading means files...", means_files)

    bic_list = []
    _, log_resp1_prefix, _, precisions_chol_prefix = _file_prefix(folder_path)
    for i, m_path in enumerate(means_files):
        n_component = x_list[i]
        try:
            log_resp1_path = f"{log_resp1_prefix}{n_component}.npy"
            precisions_chol_path = f"{precisions_chol_prefix}{n_component}.npy"
            log_resp1 = np.load(folder_path + log_resp1_path)
            precisions_chol = np.load(folder_path + precisions_chol_path)
        except:
            print(f"Cannot find {n_component} in {domain}")
            continue
        means = np.load(folder_path + m_path)
        BIC = bic(data1, means, precisions_chol, resp=np.exp(log_resp1))
        print(f"n_component: {n_component}, BIC: {BIC}")
        bic_list.append(BIC)
    plt.clf()
    plt.plot(x_list, bic_list)
    plt.savefig(graph_dir+"score_elbow.png")