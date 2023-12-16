# GaussianMixture Clustering
import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
sys.path.append("..")
from mygmm_torch import cal_diff_torch, m_setp_torch
import torch
from pycave.bayes import GaussianMixture

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

# torch.cuda.set_device(3)

##1. Set parameters
name = domain+"_"+method if method != "" else domain
# n_components = [2,10,30,50,70]
n_components = [50]
# cvgs = ["no", 80, 90, 95, 99]
cvgs = ["no"]
# random_seeds = [0, 42, 123, 987, 54321, 9999, 777, 8888, 2023, 100000]
random_seeds = [42]
# init_params = ["kmeans++", "kmeans"]
init_params = ["kmeans"]
# is_noise = ["noise", "nonoise"]
is_noise = "noise"
# dp_methods = ["rmckenna", "UCLANESL", "DPSyn", "gardn999", "PrivBayes"]
# dp_budgets = ["0.3", "1.0", "8.0"]
# dp_numbers = [1, 2, 3, 4, 5]
dp_method = "gardn999"
dp_budget = "8.0"
dp_number = 1

##2. Read data
print("Start reading data")
data1_path = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/{name}1.csv"
data2_path = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/{dp_method}/{name}_{dp_budget}_{dp_number}.csv"

data1 = pd.read_csv(data1_path, skipinitialspace=True)
# data1 = data1.loc[:, num_cols]
# data1 = data1[cat_cols]
data2 = pd.read_csv(data2_path, skipinitialspace=True)  
# data2 = data2.loc[:, num_cols]
# data2 = data2[cat_cols]
print("data1 shape", data1.shape)
print("data1 head\n", data1.head())
print("data2 shape", data2.shape)
print("data2 head\n", data2.head())

##########################################
# Attention! the data type should be int #
# But RuntimeError: cdist only supports floating-point dtypes, X1 got: Int
##########################################
data1 = torch.from_numpy(data1.values).float()
data2 = torch.from_numpy(data2.values).float()
print("torch.data1 shape", data1.shape)
print("torch.data2 shape", data2.shape)
#Apply GaussianMixture
for cvg in cvgs:
    for init_param in init_params:
        begin_time = time.time()
        for random_seed in random_seeds:
            #######################
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            #######################
            output_prefix = f"../result/{name}/{cvg}pca/{is_noise}/gpu/"
            if is_noise == "noise":
                output_prefix += f"{dp_method}_{dp_budget}_{dp_number}/"
            output_prefix += f"{init_param}/{random_seed}/"
            data_dir = output_prefix.replace("../result/", "../cluster_results/")+"data/"
            original_data_dir = data_dir.replace(f"{dp_method}_{dp_budget}_{dp_number}/", "original_data/")
            if not os.path.exists(original_data_dir):
                os.makedirs(original_data_dir)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            print("Start applying GaussianMixture")
            for n_component in n_components:
                #GMM
                #If reg_covar is not set, the vermont-diag will fail
                model = GaussianMixture(
                    num_components=n_component,
                    covariance_type='full',
                    init_strategy=init_param,
                    covariance_regularization_kmeans=1e-1,
                    covariance_regularization_gmm=6,
                    trainer_params=dict(accelerator='gpu', devices=[3])
                )
                model.fit(data1)
                score = model.score(data1)
                actual_iter = model.num_iter_
                converage = model.converged_
                # print("model.persistent_attributes", model.persistent_attributes)
                means = model.model_.means
                # print("means.shape", means.shape)
                # print("type(means)", type(means))
                covs = model.model_.covariances
                # print("covs.shape", covs.shape)
                # print("type(covs)", type(covs))
                precisions_cholesky = model.model_.precisions_cholesky
                # print("precisions_cholesky.shape", precisions_cholesky.shape)
                # print("type(precisions_cholesky)", type(precisions_cholesky))
                weights = model.model_.component_probs
                # print("weights.shape", weights.shape)
                # print("type(weights)", type(weights))

                model.save(original_data_dir)
                # torch.save(means, original_data_dir+f"means_{n_component}.pt")
                # torch.save(covs, original_data_dir+f"covs_{n_component}.pt")
                # torch.save(precisions_cholesky, original_data_dir+f"precisions_cholesky_{n_component}.pt")
                # torch.save(weights, original_data_dir+f"weights_{n_component}.pt")
                
                np.save(original_data_dir+f"means_{n_component}.npy", means.numpy())
                np.save(original_data_dir+f"covs_{n_component}.npy", covs.numpy())
                np.save(original_data_dir+f"precisions_cholesky_{n_component}.npy", precisions_cholesky.numpy())
                np.save(original_data_dir+f"weights_{n_component}.npy", weights.numpy())

                #############################################################################
                # if want to use torch to calculate m_step, may need a lot change on the code.
                # But share the memory, may not affect, what's the new np array? 
                #############################################################################

                print(f"Trying n_component={n_component}, score={score}")

        print(f"{init_param} finished, time elapsed: {time.time()-begin_time:.2f} seconds")
end_time = time.time() - begin_time
print(f"Total time elapsed: {end_time:.2f} seconds")

