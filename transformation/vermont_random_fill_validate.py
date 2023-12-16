import pandas as pd
import numpy as np

dp_methods = ["rmckenna", "UCLANESL", "DPSyn", "gardn999", "PrivBayes"]
dp_budgets = ["0.3", "1.0", "8.0"]
dp_number = 1

for dataset in ["arizona", "vermont"]:
    if dataset == "arizona":
        continue
    print(dataset)
    data_path = f"/nfsdata/tianhao/dataset/Match 3/{dataset.capitalize()}/original_data/{dataset}_nonorm.csv"
    data = pd.read_csv(data_path, skipinitialspace=True)
    print("data.shape", data.shape)
    print("data.head\n", data.head())
    cols = data.columns
    # print("cols: ", cols.values.tolist())

    for dp_method in dp_methods:
        print(f"*****{dp_method}*****")
        for dp_budget in dp_budgets:
            data_path1 = f"/nfsdata/tianhao/dataset/Match 3/{dataset.capitalize()}/{dp_method}/{dataset}_{dp_budget}_{dp_number}_nonorm.csv"
            data1 = pd.read_csv(data_path1, skipinitialspace=True)
            print("data1.shape", data1.shape)
            print("data1.head\n", data1.head())
            cols1 = data1.columns
            # print("cols1: ", cols1.values.tolist())
            diff_cols = cols.difference(cols1)
            print(f"budget={dp_budget}, diff_cols: ", diff_cols.values.tolist())
