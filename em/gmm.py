# GaussianMixture Clustering
import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
# sys.path.append("..")
from tools import create_directory
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

# read command line arguments
parser = argparse.ArgumentParser(description="GaussianMixture Clustering")

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
input_name = name+"_features"
input_path = "./dataset/" + domain + "/" + input_name + ".csv"
original_data = pd.read_csv(input_path, skipinitialspace=True, header=None)
print("original_data.shape", original_data.shape)
if method == "famd":
    minmax_scaler = MinMaxScaler()
    data = minmax_scaler.fit_transform(original_data)
else:
    data = original_data

# 2. Set parameters
# n_components_range = list(range(2, 11))+[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, \
#                                             130, 140, 150, 160, 170, 180, 190, 200]
                                            # , 300, \
                                            # 400, 500, 600, 700, 800, 900, 1000]
n_components_range = [300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
# covariance_types = ['full', 'tied', 'diag', 'spherical']
covariance_types = ['full']
output_dir = "./result/"+name+"/prediction/"
prediction_dir = output_dir+"y_pred/"
probability_dir = output_dir+"y_prob/"
if not os.path.exists(prediction_dir):
    create_directory(prediction_dir)
if not os.path.exists(probability_dir):
    create_directory(probability_dir)

# 3. Apply GaussianMixture
begin_time = time.time()
print("Start applying GaussianMixture")
for n_init in [1]:
    for covariance_type in covariance_types:
        output_path = output_dir+f"{covariance_type}.csv"
        with open(output_path, 'w') as f:
            f.write("n_components,covariance_type,time,prediction_path,probability_path\n")
        for n_components in n_components_range:
            print(f"Trying n_components={n_components}, covariance_type={covariance_type}, n_init={n_init}")
            start_time = time.time()
            # GMM
            model = GaussianMixture(n_components=n_components, \
                                    n_init=n_init, \
                                    covariance_type=covariance_type, \
                                    random_state=42
                                    )
            model.fit(data)
            elapsed_time = time.time() - start_time
            y_pred = model.predict(data)
            prediction_path = prediction_dir+f"{covariance_type}_{n_components}.csv"
            probability_path = probability_dir+f"{covariance_type}_{n_components}.csv"
            with open(output_path, 'a') as f:
                f.write(f"{n_components},{covariance_type},{elapsed_time},{prediction_path},{probability_path}\n")
            np.savetxt(prediction_path, y_pred, delimiter=",", fmt="%d")
            np.savetxt(probability_path, model.predict_proba(data), delimiter=",", fmt="%s")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"{covariance_type} finished, time elapsed: {time.time()-begin_time:.2f} seconds")
end_time = time.time() - begin_time
print(f"Total time elapsed: {end_time:.2f} seconds")

