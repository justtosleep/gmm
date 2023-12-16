import numpy as np
import pandas as pd
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country']
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital loss', 'hours-per-week']
cat_features = [col for col in names if col not in numerical_features]

save_dir = "/nfsdata/tianhao/dataset/UCLAdult/noise/"

# 1. Read data
print("Start reading data")
data_path = "/nfsdata/tianhao/dataset/UCLAdult/noise/nopca/UCLAdult_norm105_replace_0.csv"
original_data = pd.read_csv(data_path, skipinitialspace=True)
print("original_data.shape", original_data.shape)
print("original_data.head\n", original_data.head())

n_components=original_data.shape[1]
pca = PCA(n_components=n_components)
scaler = MinMaxScaler()
pca.fit(original_data)
joblib.dump(pca, f"{save_dir}nopca.pkl")

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
cvr = [i for i in zip(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)]
print("cumulative_variance_ratio", cvr)
cvgs = [80, 90, 95, 99]
idxs = []
for i in range(len(cvr)):
    if cvr[i][1] >= cvgs[len(idxs)]/100:
        idxs.append(i+1)
    if len(idxs) == len(cvgs):
        break
cvg_idx = [(cvg,idx) for cvg, idx in zip(cvgs, idxs)]
print("cvg_idx", cvg_idx)

noise_data_dict = {}
for replace_num in range(len(cat_features)+1):
    noise_data_path = f"/nfsdata/tianhao/dataset/UCLAdult/noise/nopca/UCLAdult_norm105_replace_{replace_num}.csv"
    noise_data = pd.read_csv(noise_data_path, skipinitialspace=True)
    print(f"noise_data.shape", noise_data.shape)
    print(f"noise_data.head\n", noise_data.head())
    noise_data_dict[replace_num] = noise_data

for cvg, n_component in cvg_idx:
    print(f"pca {n_component} then coverage {cvg}%")
    pca = PCA(n_components=n_component)
    pca.fit(original_data)
    save_path = f"{save_dir}{cvg}pca/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    joblib.dump(pca, f"{save_path}pca.pkl")

    for replace_num in range(len(cat_features)+1):
        noise_data = noise_data_dict[replace_num]
        pca_data = pca.transform(noise_data)
        pca_data = scaler.fit_transform(pca_data)
        print("pca_data.shape", pca_data.shape)
        print("pca_data.head\n", pca_data[0:5])
        save_path = f"{save_dir}{cvg}pca/UCLAdult_norm105_replace_{replace_num}.csv"
        pd.DataFrame(pca_data).to_csv(save_path, index=False)

