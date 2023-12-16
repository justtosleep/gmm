from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import joblib


# "arizona vermont"
name = "vermont"
print(f"dataset: {name}")
data_path = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/original_data/{name}.csv"
original_data = pd.read_csv(data_path, skipinitialspace=True)
print("original_data.shape", original_data.shape)
print("original_data.head\n", original_data.head())

#######################read noise data########################
# dp_methods = ["rmckenna", "UCLANESL", "DPSyn", "gardn999", "PrivBayes"]
dp_methods = ["rmckenna", "gardn999", "PrivBayes"]
dp_budgets = ["0.3", "1.0", "8.0"]
dp_number = 1

noise_data_dict = {}
for dp_method in dp_methods:
    noise_data_dict[dp_method] = {}
    for dp_budget in dp_budgets:
        noise_data_path = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/{dp_method}/{name}_{dp_budget}_{dp_number}.csv"
        noise_data = pd.read_csv(noise_data_path, skipinitialspace=True)
        print(f"noise_data.shape", noise_data.shape)
        print(f"noise_data.head\n", noise_data.head())
        noise_data_dict[dp_method][dp_budget] = noise_data
#######################read noise data########################

n_components=original_data.shape[1]
model_path = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/original_data/pca.pkl"
if os.path.exists(model_path):
    print("pca model exists")
    pca = joblib.load(model_path)
else:
    print("pca model not exists")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pca = PCA(n_components=n_components)
    pca.fit(original_data)
    joblib.dump(pca, model_path)

scaler = MinMaxScaler()
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
cvr = [i for i in zip(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)]
print("cumulative_variance_ratio", cvr)

# cvgs = [80, 90, 95, 99]
cvgs = [90]
idxs = []
for i in range(len(cvr)):
    if cvr[i][1] >= cvgs[len(idxs)]/100:
        idxs.append(i+1)
    if len(idxs) == len(cvgs):
        break
cvg_idx = [(cvg,idx) for cvg, idx in zip(cvgs, idxs)]
print("cvg_idx", cvg_idx)

for cvg, n_component in cvg_idx:
    print(f"pca {n_component} then coverage {cvg}%")
    sub_model_dir = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/original_data/{cvg}pca/"
    sub_model_path = f"{sub_model_dir}pca.pkl"
    if os.path.exists(sub_model_path):
        print(f"pca coverage {cvg}% model exists")
        pca = joblib.load(sub_model_path)
    else:
        print(f"pca coverage {cvg}% model not exists")
        os.makedirs(sub_model_dir, exist_ok=True)
        pca = PCA(n_components=n_component)
        pca.fit(original_data)
        joblib.dump(pca, f"{sub_model_path}")
        pca_original_data = pca.transform(original_data)
        pca_original_data = scaler.fit_transform(pca_original_data)
        print(f"pca_original_data.shape", pca_original_data.shape)
        print(f"pca_original_data.head\n", pca_original_data[:5])
        save_pcadata_path = f"{sub_model_dir}{name}.csv"
        pca_original_data_df = pd.DataFrame(pca_original_data)
        pca_original_data_df.to_csv(save_pcadata_path, index=False)

########################pca noise data########################
    for dp_method in dp_methods:
        for dp_budget in dp_budgets:
            noise_data = noise_data_dict[dp_method][dp_budget]
            pca_data = pca.transform(noise_data)
            pca_data = scaler.fit_transform(pca_data)
            print(f"pca_data.shape", pca_data.shape)
            print(f"pca_data.head\n", pca_data[:5])
            save_pcadata_dir = f"/nfsdata/tianhao/dataset/Match 3/{name.capitalize()}/{dp_method}/{cvg}pca/"
            save_pcadata_path = f"{save_pcadata_dir}{name}_{dp_budget}_{dp_number}.csv"
            os.makedirs(save_pcadata_dir, exist_ok=True)
            pca_data_df = pd.DataFrame(pca_data)
            pca_data_df.to_csv(save_pcadata_path, index=False)
########################pca noise data########################

        
