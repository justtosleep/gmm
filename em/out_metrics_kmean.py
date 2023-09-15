import os
import numpy as np
import pandas as pd
import argparse
from tools import create_directory, output_metrics, pred2prob, calculate_centers


# read command line arguments
parser = argparse.ArgumentParser(description="Kmeans Clustering")

# add positional arguments
parser.add_argument("dataset", type=str, help="name of dataset")

# add optional arguments
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
method = args.method

# 1. Read data
print("Start reading data")
name = domain+"_"+method if method != "" else domain
input_dir = "./dataset/"+domain+"/"
data_path = input_dir + domain + "_features.csv"
data = pd.read_csv(data_path, skipinitialspace=True, header=None)
data = np.array(data)
print("data.shape", data.shape)

#2. Set parameters
best_results = {
    'components': 0,
    'mse': np.inf,
    'mae': np.inf,
    'time': 0
}

# result_dir = "./metrics/" + name + "_kmean/measure"
result_dir = "./metrics/" + name + "/measure"
output_dir = result_dir + "/"
if not os.path.exists(output_dir):
    create_directory(output_dir)
output_best_path = output_dir+"best.csv"
with open(output_best_path, 'a') as f:
    f.write("n_components,mse,mae,time\n")

#3. Output metrics
# n_init = 1
# pred_dir = "./result/" + name + "_kmean/prediction/"
# pred_path = pred_dir+f"kmean.csv"
pred_dir = "./result/" + name + "/prediction/"
pred_path = pred_dir+f"kfreq.csv"
pred = pd.read_csv(pred_path, skipinitialspace=True)

if method == "mix":
    n_components = pred['classes'].values
else:
    n_components = pred['n_components'].values
times = pred['time'].values
prediction_paths = pred['prediction_path'].values

# output_path = output_dir+f"kmean.csv"
output_path = output_dir+f"kfreq.csv"
with open(output_path, 'w') as f:
    f.write("n_components,mse,mae,time\n")

for i, prediction_path in enumerate(prediction_paths):
    n_component = n_components[i]
    elapsed_time = times[i]
    print(f"reading:  n_components={n_component}")
    y_pred = np.loadtxt(prediction_path, delimiter=",").astype(np.int32)

    centers = calculate_centers(data, y_pred)
    # print("centers.shape", centers.shape)

    mse, mae = output_metrics(y_pred, pred2prob(y_pred), data, centers)
    with open(output_path, 'a') as f:
        f.write(f"{n_component},{mse},{mae},{elapsed_time}\n")
    if mse < best_results['mse']:
        best_results['components'] = n_component
        best_results['mse'] = mse
        best_results['mae'] = mae
        best_results['time'] = elapsed_time

print(f"Best results so far: {best_results}")
with open(output_best_path, 'a') as f:
    f.write(f"{best_results['components']},{best_results['mse']},{best_results['mae']},{best_results['time']}\n")


# # 4. Output one prediction
# pred_dir = "./result/"+name+"/prediction/y_pred/"
# pred_path = pred_dir+name+"_spherical_1000_pred.csv"
# y_pred = np.loadtxt(pred_path, delimiter=",").astype(np.int32)
# print("y_pred.shape", y_pred.shape)
# print("data.shape", data.shape)
# centers = calculate_centers(data, y_pred)
# mse, mae, ari, nmi, p, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg = output_metrics(labels, y_pred, data, centers)
# print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {p:.4f}, Silhouette: {silhouette_avg:.4f}, Calinski-Harabasz: {calinski_harabasz_avg:.4f}, Davies-Bouldin: {davies_bouldin_avg:.4f}")
