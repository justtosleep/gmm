import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy import linalg
from scipy.special import logsumexp
from myvil import diff_seed_noise_torch
from myvil import diff_seed_noise
import torch

torch.cuda.set_device(3)

def cal_avg_var_median(data_list):
    data_list = np.array(data_list)
    print("data_list shape: ", data_list.shape)
    avg = np.average(data_list, axis=0)
    var = np.var(data_list, axis=0)
    median = np.median(data_list, axis=0)
    print("avg: ", avg)
    print("var: ", var)
    print("median: ", median)
    return avg, var, median

def box_plot(box_data, org_data, x_labels, filename, graph_dir, dp_method):
    fontsize = 14
    plt.clf()
    medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                        markerfacecolor='blue')
    x_ticks = np.arange(1, len(x_labels)+1)
    plt.boxplot(
        box_data,
        labels=[str(x) for x in x_ticks],
        showfliers=False,
        meanprops=meanpointprops, 
        meanline=False,
        showmeans=True,
        medianprops=medianprops
        )
    plt.xticks(x_ticks, [str(x) for x in x_labels])
    # set legend
    mean_legend = plt.Line2D([0], [0], marker='D', markeredgecolor='black',
                        markerfacecolor='blue', label='noise data avg')
    median_legend = plt.Line2D([0], [0], linestyle='-.', linewidth=2.5, color='firebrick', label='noise data median')
    org_data_legend = plt.Line2D([0], [0], marker='o', color='blue', label='orginal data avg')
    plt.legend(handles=[mean_legend, median_legend])
    plt.title(
        f"Compare {filename} update abs on original data and dp noise data \n({dp_method} | kmeans | number of clusters: 50 | 10 seeds)", 
        fontsize=fontsize
        )
    plt.xlabel('Buget', fontsize=fontsize)
    plt.ylabel(f"log10({filename} update abs)", fontsize=fontsize)
    plt.savefig(f"{graph_dir}kmeans_{filename}.png")

    plt.legend(handles=[mean_legend, median_legend, org_data_legend])
    plt.scatter(x_ticks, org_data, marker='o', color='blue', label='orginal data avg')
    plt.savefig(f"{graph_dir}org_kmeans_{filename}.png")

begin_time = time.time()
# read command line arguments
parser = argparse.ArgumentParser(description="GaussianMixture Clustering metrics")

parser.add_argument("--dataset", type=str, default="arizona", help="name of dataset")
# parser.add_argument("--data_path", type=str, default="Match 3/Arizona/arizona.csv", help="path of dataset")
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
# data_path = args.data_path
method = args.method
print("domain: ", domain)
# print("data path: ", data_path)
print("method: ", method)

#####################1. Set parameters#####################
# cvgs = ["no", 80, 90, 95, 99]
cvgs = [90]
n_components = 50
init_param = "kmeans"

# dp_methods = ["rmckenna", "UCLANESL", "DPSyn", "gardn999", "PrivBayes"]
# dp_methods = ["rmckenna", "DPSyn", "PrivBayes"]
dp_method = "rmckenna"
dp_budgets = ["0.3", "1.0", "8.0"]
# dp_numbers = [1, 2, 3, 4, 5]
dp_number = 1
#####################1. Set parameters#####################

name = domain+"_"+method if method != "" else domain
for cvg in cvgs:
    new_time = time.time()
    print(f"**********{domain} {cvg}% pca**********")
    #####################2. Read data#####################
    data_dir = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/"
    data1_path = f"{data_dir}original_data/{cvg}pca/{name}.csv"
    data1 = pd.read_csv(data1_path, skipinitialspace=True)
    print("data1 shape", data1.shape)
    print("data1 head\n", data1.head())

    noise_data_dict = {}
    for dp_budget in dp_budgets:
        noise_data_path = f"{data_dir}{dp_method}/{cvg}pca/{name}_{dp_budget}_{dp_number}.csv"
        noise_data = pd.read_csv(noise_data_path, skipinitialspace=True)
        print(f"{dp_method}_{dp_budget}_{dp_number} noise_data.shape", noise_data.shape)
        print(f"noise_data.head\n", noise_data.head())
        noise_data_dict[dp_budget] = noise_data
    #####################2. Read data#####################

    graph_dir = f"../result/{name}/{cvg}pca/noise/{dp_method}/graph/"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    original_data_dir = f"/nfsdata/tianhao/cluster_results/{name}/{cvg}pca/noise/original_data/"

    begin_time = time.time()

    m2_boxplot = []
    c2_boxplot = []
    m1_avgs = []
    c1_avgs = []
    for dp_budget in dp_budgets:
        print(f"****** {dp_method}_{dp_budget}_{dp_number}******")
        data2 = noise_data_dict[dp_budget]

        result_dir = f"/nfsdata/tianhao/cluster_results/{name}/{cvg}pca/noise/{dp_method}_{dp_budget}_{dp_number}/"
        random_seeds = [x for x in os.listdir(f"{result_dir}/{init_param}/")]
        print(f"init_param: {init_param}, random_seed: {random_seeds}")
        
        m1_list = []
        m2_list = []
        c1_list = []
        c2_list = []
        for random_seed in random_seeds:
            org_result_dir = f"{original_data_dir}{init_param}/{random_seed}/data/"
            noise_result_dir = f"{result_dir}{init_param}/{random_seed}/data/"
            n_components_list = [50]
            print(f"random_seed: {random_seed}")
            _, means_diff1_list, covariances_diff1_list, means_diff2_list, covariances_diff2_list = diff_seed_noise_torch(org_result_dir, noise_result_dir, data1, data2, name, n_components_list)
            # _, means_diff1_list, covariances_diff1_list, means_diff2_list, covariances_diff2_list = diff_seed_noise(org_result_dir, noise_result_dir, data1, data2, name, n_components_list)
            m1_list.append(means_diff1_list[0])
            m2_list.append(means_diff2_list[0])
            c1_list.append(covariances_diff1_list[0])
            c2_list.append(covariances_diff2_list[0])
        m1_avg, _, _ = cal_avg_var_median(m1_list)
        c1_avg, _, _ = cal_avg_var_median(c1_list)

        m1_avgs.append(np.log10(m1_avg))
        c1_avgs.append(np.log10(c1_avg))
        m2_boxplot.append(np.log10(m2_list))
        c2_boxplot.append(np.log10(c2_list))

    # 4. Plot
    plt.clf()
    width = 10
    height = 10
    plt.figure(figsize=(width, height))
    x_ticks = [float(x) for x in dp_budgets]
    print("x_ticks: ", x_ticks)
    print("m2_boxplot: ", m2_boxplot)
    print("c2_boxplot: ", c2_boxplot)
    print("m1_avgs: ", m1_avgs)
    print("c1_avgs: ", c1_avgs)
    box_plot(m2_boxplot, m1_avgs, x_ticks, "means", graph_dir, dp_method)
    box_plot(c2_boxplot, c1_avgs, x_ticks, "covariances", graph_dir, dp_method)

    print(f"cvgs {cvg}% finished, time elapsed: {time.time()-new_time:.2f} seconds")
print("total time: ", time.time()-begin_time)
