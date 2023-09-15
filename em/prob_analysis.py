import os
import time
import argparse
import numpy as np
import pandas as pd
from tools import create_directory


#read command line arguments
parser = argparse.ArgumentParser(description="GaussianMixture Clustering")

#add positional arguments
parser.add_argument("dataset", type=str, help="name of dataset")
#add optional arguments
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
method = args.method

#1. Set parameters
print("Start reading data")
name = domain+"_"+method if method != "" else domain
pred_dir = "./result/" + name + "/prediction/"
# covariance_types = ['full', 'tied', 'diag', 'spherical']
covariance_types = ['diag']

output_dir = "./metrics/"+name+"/prob_dist/"
if not os.path.exists(output_dir):
    create_directory(output_dir)
output_best_path = output_dir+"max_diff.csv"
with open(output_best_path, 'a') as f:
    f.write("covariance_type,n_components,p90-p10\n")

#2. Output prob metrics
for covariance_type in covariance_types:
    print(f"start read {covariance_type}")
    
    pred_path = pred_dir+f"{covariance_type}.csv"
    pred = pd.read_csv(pred_path, skipinitialspace=True)
    if method == "mix":
        n_components = pred['classes'].values
    else:
        n_components = pred['n_components'].values
    probability_paths = pred['probability_path'].values
    max_diff = {
        "n_components": 0,
        "diff": 0,
    }
    output_prob_path = output_dir+f"{covariance_type}.csv"
    with open(output_prob_path, 'w') as f:
        f.write("n_components,>0.9,>0.75,>0.5,>0.25,>0.1\n")
    for i, probability_path in enumerate(probability_paths):
        y_prob = np.loadtxt(probability_paths[i], delimiter=",").astype(np.float32)
        if i == 0:
            print("data.shape", y_prob.shape[0])
        p90 = y_prob>0.9
        p75 = y_prob>0.75
        p50 = y_prob>0.5
        p25 = y_prob>0.25
        p10 = y_prob>0.1
        diff = np.sum(p10)-np.sum(p90)
        if diff > max_diff["diff"]:
            max_diff["n_components"] = n_components[i]
            max_diff["diff"] = diff
        with open(output_prob_path, 'a') as f:
            f.write(f"{n_components[i]},{np.sum(p90)},{np.sum(p75)},{np.sum(p50)},{np.sum(p25)},{np.sum(p10)}\n")
        # print(f"n_components: {n_components[i]}", f"numbers: >0.9: {np.sum(p90)}", f">0.75: {np.sum(p75)}", f">0.5: {np.sum(p50)}", f">0.25: {np.sum(p25)}", f">0.1: {np.sum(p10)}")
        # print(f"[normal]>0.9: {np.sum(p90)*0.9}", f">=0.75: {np.sum(p75)*0.75}", f">=0.5: {np.sum(p50)*0.5}", f">=0.25: {np.sum(p25)*0.25}", f">=0.1: {np.sum(p10)*0.1}")
        # print(f"ratio: >0.9: {np.sum(p90)/y_prob.shape[0]}", f">0.75: {np.sum(p75)/y_prob.shape[0]}", f">0.5: {np.sum(p50)/y_prob.shape[0]}", f">0.25: {np.sum(p25)/y_prob.shape[0]}", f">0.1: {np.sum(p10)/y_prob.shape[0]}")
    with open(output_best_path, 'a') as f:
        f.write(f"{covariance_type},{max_diff['n_components']},{max_diff['diff']}\n")
    print(f"covariance_type: {covariance_type}", f"n_components: {max_diff['n_components']}", f"max_diff: {max_diff['diff']}")
# print("Start reading data")
# domin = "UCLAdult" #  "abalone""covtype"
# name = domin+"_famd"
# input_path = '../../dataset/adult/adult.data'
# names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country', 'result']
# original_data = pd.read_csv(input_path, names = names, skipinitialspace=True).iloc[1:]
# print("original_data.shape", original_data.shape)
# print("original_data.head", original_data.head())

# # clean_path = "data/" + name + "/" + domin + "_clean.csv"
# # clean_data = pd.read_csv(clean_path, skipinitialspace=True)
# # print("clean_data.shape", clean_data.shape)
# # print("clean_data.head", clean_data.head())

# # age1 = np.array(original_data['age'])
# # age2 = np.array(clean_data['age'])
# # equal = np.equal(age1, age2)
# # print("equal", sum(equal))

# sparse_path = '../sparse/dataset/UCLAdult/UCLAdult_features.csv'
# sparse_data = pd.read_csv(sparse_path, skipinitialspace=True, header=None).iloc[2:]
# print("sparse_data.shape", sparse_data.shape)
# print("sparse_data.head", sparse_data.head())

# # famd_path = "data/" + name + "/" + name + "_clean.csv"
# # famd_data = pd.read_csv(famd_path, skipinitialspace=True, header=None)
# # print("famd_data.shape", famd_data.shape)
# # print("famd_data.head", famd_data.head())
