# Kmeans Clustering
import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
from tools import create_directory
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


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
n_components_range = list(range(2, 11))+[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, \
                                            130, 140, 150, 160, 170, 180, 190, 200]
                                            # , 300, \
                                            # 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, \
                                            # 3000, 4000]
# n_components_range = [3000, 4000]
output_dir = "./result/"+name+"_kmean/prediction/"
prediction_dir = output_dir+"y_pred/"
if not os.path.exists(prediction_dir):
    create_directory(prediction_dir)

# 3. Apply GaussianMixture
begin_time = time.time()
print("Start applying GaussianMixture")
output_path = output_dir+f"kmean.csv"
with open(output_path, 'w') as f:
    f.write("n_components,time,prediction_path\n")
for n_components in n_components_range:
    print(f"Trying n_components={n_components}")
    start_time = time.time()
    # Kmeans
    model = KMeans( n_clusters=n_components, \
                    n_init=1, \
                    max_iter=1000, \
                    random_state=42)
    model.fit(data)
    elapsed_time = time.time() - start_time
    y_pred = model.predict(data)
    prediction_path = prediction_dir+f"{n_components}.csv"
    with open(output_path, 'a') as f:
        f.write(f"{n_components},{elapsed_time},{prediction_path}\n")
    np.savetxt(prediction_path, y_pred, delimiter=",", fmt="%d")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
print(f"finished, time elapsed: {time.time()-begin_time:.2f} seconds")
end_time = time.time() - begin_time
print(f"Total time elapsed: {end_time:.2f} seconds")


