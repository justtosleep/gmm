# nohup python mse_sample12.py --dataset UCLAdult --data_path UCLAdult/UCLAdult_sample1.data  --method sample1 > "../log/sample12_metric.out" 2>&1 &
import os
import numpy as np
import pandas as pd
import argparse
from tools import create_directory, output_metrics, pred2prob, calculate_centers

# read command line arguments
parser = argparse.ArgumentParser(description="Calculate the metrics of dataset.")

# add positional arguments

# add optional arguments
parser.add_argument("--dataset", type=str, help="name of dataset")
parser.add_argument("--data_path", type=str, help="path of dataset")
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
data_path = args.data_path
method = args.method

# 1. Read data
print("Start reading data")
name = domain+"_"+method if method != "" else domain
covariance_types = ['full', 'tied', 'diag', 'spherical']
# covariance_types = ['full']
if domain == "UCLAdult":
    if method == "sample1":
        data_path = "UCLAdult/UCLAdult_sample1.data"
    elif method == "sample2":
        data_path = "UCLAdult/UCLAdult_sample2.data"
    else:
        data_path = "UCLAdult/UCLAdult_norm105.data"
data_path = "../dataset/" + data_path
data = pd.read_csv(data_path, skipinitialspace=True)
data = np.array(data)
print("data.shape", data.shape)


domain = "UCLAdult"
data1_path = "../dataset/UCLAdult/UCLAdult_sample1.data"
method1 = "sample1"
data2_path = "../dataset/UCLAdult/UCLAdult_sample2.data"
method2 = "sample2"
covariance_types = ['full', 'tied', 'diag', 'spherical']

data1 = pd.read_csv(data1_path, skipinitialspace=True)
print("data1 shape", data1.shape)
data2 = pd.read_csv(data2_path, skipinitialspace=True)
data2 = np.array(data2)
print("data2 shape", data2.shape)

#prediction metrics
output_dir = "../metrics/UCLAdult_sample1/measure_sample2/"
if not os.path.exists(output_dir):
    create_directory(output_dir)

#3. Output metrics
# n_init = 1
pred_dir = "../result/" + name + "/prediction/"
for covariance_type in covariance_types:
    pred_path = pred_dir+f"{covariance_type}.csv"
    pred = pd.read_csv(pred_path, skipinitialspace=True)

    if method == "mix":
        n_components = pred['classes'].values
    else:
        n_components = pred['n_components'].values
    prediction_paths = pred['prediction_path'].values
    # probability_paths = pred['probability_path'].values

    output_path = output_dir+f"{covariance_type}.csv"
    with open(output_path, 'w') as f:
        f.write("n_components,mse,mae\n")
    

    for i, prediction_path in enumerate(prediction_paths):
        n_component = n_components[i]
        print(f"reading: covariance_type={covariance_type}, n_components={n_component}")
        y_pred = np.loadtxt(prediction_path, delimiter=",").astype(np.int32)

        centers_dict = calculate_centers(data, y_pred)
        if centers_dict['empty'] != []:
            print("empty cluster exists")
            continue
        centers = np.array(centers_dict['centers'])
        mse_distances = np.sum((data2[:, np.newaxis] - centers)**2, axis=2)
        mae_distances = np.sum(np.abs(data2[:, np.newaxis] - centers), axis=2)
        min_mse_distances = np.min(mse_distances, axis=1)
        min_mae_distances = np.min(mae_distances, axis=1)
        mse = np.sum(min_mse_distances)/data2.shape[0]
        mae = np.sum(min_mae_distances)/data2.shape[0]

        with open(output_path, 'a') as f:
            f.write(f"{n_component},{mse},{mae}\n")

# # 4. Output one prediction
# pred_dir = "./result/"+name+"/prediction/y_pred/"
# pred_path = pred_dir+name+"_spherical_1000_pred.csv"
# y_pred = np.loadtxt(pred_path, delimiter=",").astype(np.int32)
# print("y_pred.shape", y_pred.shape)
# print("data.shape", data.shape)
# centers = calculate_centers(data, y_pred)
# mse, mae, ari, nmi, p, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg = output_metrics(labels, y_pred, data, centers)
# print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {p:.4f}, Silhouette: {silhouette_avg:.4f}, Calinski-Harabasz: {calinski_harabasz_avg:.4f}, Davies-Bouldin: {davies_bouldin_avg:.4f}")
