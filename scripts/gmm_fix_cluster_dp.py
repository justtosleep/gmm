import os
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import joblib
import argparse
from mygmm import cal_diff, m_setp


def cal_log_resp(resp):
    resp += 10 * np.finfo(resp.dtype).eps
    resp /= resp.sum()
    return np.log(resp)

def save_data_info(resp, labels, data_dir, n_component, num):
    log_resp = cal_log_resp(resp)
    np.save(data_dir+f"class_proportions{num}_log_resp_{n_component}.npy", log_resp)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    np.save(data_dir+f"class_proportions{num}_unique_labels_{n_component}.npy", unique_labels)
    np.save(data_dir+f"class_proportions{num}_label_counts_{n_component}.npy", label_counts)

    
begin_time = time.time()
# read command line arguments
parser = argparse.ArgumentParser(description="GaussianMixture Clustering metrics")

# add arguments
parser.add_argument("--dataset", type=str, default="arizona", help="name of dataset")
parser.add_argument("--data_path", type=str, default="Match 3/Arizona/arizona.csv", help="path of dataset")
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
# data_path = args.data_path
method = args.method
print("domain: ", domain)
# print("data path: ", data_path)
print("method: ", method)

#####################1. Set parameters#####################
random_seeds = [0, 42, 123, 987, 54321, 9999, 777, 8888, 2023, 100000]
# n_components = [2,10,30,50,70]
n_components = 50
# init_params = ["k-means++", "kmeans"]
init_param = "kmeans"
# dp_methods = ["rmckenna", "UCLANESL", "DPSyn", "gardn999", "PrivBayes"]
# dp_methods = ["rmckenna", "DPSyn", "PrivBayes"]
dp_method = "rmckenna"
dp_budgets = ["0.3", "1.0", "8.0"]
dp_number = 1
# cvgs=[20,50,90,99,"no"]
cvgs=[90]
#####################1. Set parameters#####################

name = domain+"_"+method if method != "" else domain
for cvg in cvgs:
    print(f"**********{domain} {cvg}% pca**********")
    #####################2. Read data#####################
    print("Start reading data")
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

    #####################3. Apply GaussianMixture#####################
    original_data_dir_prefix = f"../cluster_results/{name}/{cvg}pca/noise/original_data/{init_param}/"
    for random_seed in random_seeds:
        new_time = time.time()
        
        original_data_dir = original_data_dir_prefix+f"{random_seed}/data/"
        if not os.path.exists(original_data_dir):
            os.makedirs(original_data_dir)
            
        model_path = (original_data_dir_prefix.replace("../", "/nfsdata/tianhao/") + 
              f"{random_seed}/data/model_{n_components}.pkl")
        if os.path.exists(model_path):
            print("model exists")
            model = joblib.load(model_path)
        else:
            print("model not exists")
            model = GaussianMixture(n_components=n_components, \
                                    random_state=random_seed, \
                                    init_params=init_param, \
                                    max_iter=150, \
                                    n_init=1, \
                                    )
            model.fit(data1)
            joblib.dump(model, f"{original_data_dir}model_{n_components}.pkl")
        ####################save model info####################
        model_info_path = (original_data_dir_prefix.replace("../", "/nfsdata/tianhao/") + 
              f"{random_seed}/data/means_{n_components}.npy")
        if not os.path.exists(model_info_path):
            np.save(original_data_dir+f"means_{n_components}.npy", model.means_)
            np.save(original_data_dir+f"covariances_{n_components}.npy", model.covariances_)
            np.save(original_data_dir+f"precisions_chol_{n_components}.npy", model.precisions_cholesky_)
            np.save(original_data_dir+f"weights_{n_components}.npy", model.weights_)

            resp = model.predict_proba(data1)
            labels = model.predict(data1)
            save_data_info(resp, labels, original_data_dir, n_components, 1)
        ####################save model info####################

        print(f"random_seed {random_seed} training time: ", time.time()-new_time)
        score = model.score(data1)

        # resp2, means2, covariances2, precisions_chol2 = m_setp(data2, log_resp, covariance_type)
        # l1_means_diff2, l1_covariances_diff2 = cal_diff(means, covariances, means2, covariances2)
        # ####################save data info####################
        for i,dp_budget in enumerate(dp_budgets):
            noise_result_dir = f"../cluster_results/{name}/{cvg}pca/noise/{dp_method}_{dp_budget}_{dp_number}/{init_param}/{random_seed}/data/"
            if not os.path.exists(noise_result_dir):
                os.makedirs(noise_result_dir)
            noise_data = noise_data_dict[dp_budget]

            resp = model.predict_proba(noise_data)
            labels = model.predict(noise_data)
            save_data_info(resp, labels,  noise_result_dir, n_components, 2)
        # ####################save data info####################

        print(f"random_seed {random_seed} finished, time elapsed: {time.time()-new_time:.2f} seconds")
    end_time = time.time() - begin_time
    print(f"Total time elapsed: {end_time:.2f} seconds")

