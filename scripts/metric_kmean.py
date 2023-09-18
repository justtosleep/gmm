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
if domain == "UCLAdult":
    data_path = "UCLAdult/UCLAdult_norm105.data"
data_path = "../dataset/" + data_path
data = pd.read_csv(data_path, skipinitialspace=True)
data = np.array(data)
print("data.shape", data.shape)

#2. Set parameters
output_dir = "../metrics/" + name + "/measure_kmean/"
if not os.path.exists(output_dir):
    create_directory(output_dir)

best_results = {
    'components': 0,
    'mse': np.inf,
    'mae': np.inf,
    'time': 0
}

#3. Output metrics
pred_dir = "./result/" + name + "/prediction_kmean/"
pred_path = pred_dir+f"kmean.csv"
pred = pd.read_csv(pred_path, skipinitialspace=True)

if method == "mix":
    n_components = pred['classes'].values
else:
    n_components = pred['n_components'].values
prediction_paths = pred['prediction_path'].values
times = pred['time'].values
output_path = output_dir+f"kmean.csv"
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
