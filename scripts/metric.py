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

#2. Set parameters
best_results = {
    'components': 0,
    'covariance_type': None,
    'mse': np.inf,
    'mae': np.inf,
    'time': 0
}
best_results_prob = {
    'components': 0,
    'covariance_type': None,
    'mse': np.inf,
    'mae': np.inf,
    'time': 0
}
max_diff = {
    "n_components": 0,
    "diff": 0,
}
#prediction metrics
result_dir = "../metrics/" + name + "/measure"
output_dir = result_dir + "/"
if not os.path.exists(output_dir):
    create_directory(output_dir)
output_best_path = output_dir+"best.csv"
with open(output_best_path, 'a') as f:
    f.write("n_components,covariance_type,mse,mae,time\n")

#probability weight metrics
outprob_dir = result_dir + "_prob/"
if not os.path.exists(outprob_dir):
    create_directory(outprob_dir)
outprob_best_path = outprob_dir+"best.csv"
with open(outprob_best_path, 'a') as f:
    f.write("n_components,covariance_type,mse,mae,time\n")

#probability analysis
probdist_dir = "../metrics/" + name + "/prob_dist/"
if not os.path.exists(probdist_dir):
    create_directory(probdist_dir)
probdist_best_path = probdist_dir+"max_diff.csv"
with open(probdist_best_path, 'a') as f:
    f.write("covariance_type,n_components,p90-p10\n")

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
    times = pred['time'].values
    prediction_paths = pred['prediction_path'].values
    probability_paths = pred['probability_path'].values

    output_path = output_dir+f"{covariance_type}.csv"
    with open(output_path, 'w') as f:
        f.write("n_components,mse,mae,time\n")
    outprob_path = outprob_dir+f"{covariance_type}.csv"
    with open(outprob_path, 'w') as f:
        f.write("n_components,mse,mae\n")
    probdist_path = probdist_dir+f"{covariance_type}.csv"
    with open(probdist_path, 'w') as f:
        f.write("n_components,>0.9,>0.75,>0.5,>0.25,>0.1\n")

    for i, prediction_path in enumerate(prediction_paths):
        n_component = n_components[i]
        elapsed_time = times[i]
        print(f"reading: covariance_type={covariance_type}, n_components={n_component}")
        y_pred = np.loadtxt(prediction_path, delimiter=",").astype(np.int32)
        y_prob = np.loadtxt(probability_paths[i], delimiter=",").astype(np.float32)

        centers = calculate_centers(data, y_pred)
        # print("centers.shape", centers.shape)

        #prediction metrics
        mse, mae = output_metrics(y_pred, pred2prob(y_pred), data, centers)
        with open(output_path, 'a') as f:
            f.write(f"{n_component},{mse},{mae},{elapsed_time}\n")
        if mse < best_results['mse']:
            best_results['components'] = n_component
            best_results['covariance_type'] = covariance_type
            best_results['mse'] = mse
            best_results['mae'] = mae
            best_results['time'] = elapsed_time

        #probability weight metrics
        mse, mae = output_metrics(y_pred, y_prob, data, centers)
        with open(outprob_path, 'a') as f:
            f.write(f"{n_component},{mse},{mae}\n")
        if mse < best_results_prob['mse']:
            best_results_prob['components'] = n_component
            best_results_prob['covariance_type'] = covariance_type
            best_results_prob['mse'] = mse
            best_results_prob['mae'] = mae
            best_results_prob['time'] = elapsed_time
        
        #probability analysis
        p90 = y_prob>0.9
        p75 = y_prob>0.75
        p50 = y_prob>0.5
        p25 = y_prob>0.25
        p10 = y_prob>0.1
        diff = np.sum(p10)-np.sum(p90)
        if diff > max_diff["diff"]:
            max_diff["n_components"] = n_component
            max_diff["diff"] = diff
        with open(probdist_path, 'a') as f:
            f.write(f"{n_component},{np.sum(p90)},{np.sum(p75)},{np.sum(p50)},{np.sum(p25)},{np.sum(p10)}\n")

    print(f"Best results so far: {best_results}")
    with open(output_best_path, 'a') as f:
        f.write(f"{best_results['components']},{best_results['covariance_type']},{best_results['mse']},{best_results['mae']},{best_results['time']}\n")
    with open(outprob_best_path, 'a') as f:
        f.write(f"{best_results_prob['components']},{best_results_prob['covariance_type']},{best_results_prob['mse']},{best_results_prob['mae']},{best_results_prob['time']}\n")
    with open(probdist_best_path, 'a') as f:
        f.write(f"{covariance_type},{max_diff['n_components']},{max_diff['diff']}\n")

# # 4. Output one prediction
# pred_dir = "./result/"+name+"/prediction/y_pred/"
# pred_path = pred_dir+name+"_spherical_1000_pred.csv"
# y_pred = np.loadtxt(pred_path, delimiter=",").astype(np.int32)
# print("y_pred.shape", y_pred.shape)
# print("data.shape", data.shape)
# centers = calculate_centers(data, y_pred)
# mse, mae, ari, nmi, p, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg = output_metrics(labels, y_pred, data, centers)
# print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {p:.4f}, Silhouette: {silhouette_avg:.4f}, Calinski-Harabasz: {calinski_harabasz_avg:.4f}, Davies-Bouldin: {davies_bouldin_avg:.4f}")
