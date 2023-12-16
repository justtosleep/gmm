import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random


names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country']
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital loss', 'hours-per-week']
cat_features = [col for col in names if col not in numerical_features]

save_dir = "/nfsdata/tianhao/dataset/UCLAdult/noise/nopca/"

def norm105(data, filename=""):
    minmax_scaler = MinMaxScaler()
    numerical_data = minmax_scaler.fit_transform(data[numerical_features])
    print("numerical_data.shape", numerical_data.shape)
    print("numerical_data.head", numerical_data[0:5])

    one_hot_data = pd.get_dummies(data[cat_features])
    print("one_hot_data.shape", one_hot_data.shape)
    print("one_hot_data.head", one_hot_data.head())
    numerical_data = pd.DataFrame(numerical_data)
    
    new_data = pd.concat([numerical_data, one_hot_data], axis=1)
    print("new_data.shape", new_data.shape)
    print("new_data.head", new_data.head())

    save_path = save_dir+f"{filename}"
    new_data.to_csv(save_path, index=False)

# 1. Read data
print("Start reading data")
data_path = "/nfsdata/tianhao/dataset/UCLAdult/UCLAdult.data"
original_data = pd.read_csv(data_path, skipinitialspace=True)
print("original_data.shape", original_data.shape)
print("original_data.head\n", original_data.head())

norm105(original_data, "UCLAdult_norm105_replace_0.csv")

replace_nums = len(cat_features)
for replace_num in range(replace_nums):
    noise_data_path = f"/nfsdata/tianhao/dataset/UCLAdult/noise/UCLAdult_replace_{replace_num+1}.csv"
    noise_data = pd.read_csv(noise_data_path, skipinitialspace=True)
    print(f"noise_data.shape", noise_data.shape)
    print(f"noise_data.head\n", noise_data.head())
    norm105(noise_data, f"UCLAdult_norm105_replace_{replace_num+1}.csv")
