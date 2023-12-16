import pandas as pd
import numpy as np
import random
random.seed(42)

navalues_path = "/nfsdata/tianhao/dataset/Match 3/NAvalues.txt"
with open(navalues_path, 'r') as file:
    lines = file.readlines()
data_dict = {}
for line in lines:
    line = line.strip()
    line = line.rstrip(',')
    key, value = line.split(":")
    key = eval(key)
    value = eval(value)
    data_dict[key] = value
# print("Finish reading NAvalues.txt")
# print("data_dict: \n", data_dict)

# # dp_methods = ["rmckenna", "DPSyn", "gardn999", "PrivBayes"]
# dp_methods = ["DPSyn", "gardn999", "PrivBayes"]
# dp_budgets = ["0.3", "1.0", "8.0"]
# # dp_budgets = ["0.3"]
# dp_number = 1

# for dp_method in dp_methods:
#     for dp_budget in dp_budgets:
#         for key in data_dict.keys():
#             # if key == "vermont":
#             #     continue
#             print("\nkey: ", key)
#             nas = data_dict[key]
#             print("value length: ", len(nas))
#             if key == "arizona":
#                 key_abbreviation = "az"
#             elif key == "vermont":
#                 key_abbreviation = "vt"
#             # data_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{key}0.csv"
#             data_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{dp_method}/out_{key_abbreviation}_{dp_budget}_{dp_number}.csv"
#             data = pd.read_csv(data_path, skipinitialspace=True).astype(np.int64)
#             col_names = data.columns.values.tolist()
#             n_samples, n_features = data.shape
#             print("data shape: ", data.shape)
#             print("data head: \n", data.head())
#             print("col_names: \n", col_names)

#             na_colnames = []
#             threshold = 0.4
#             for i, na in enumerate(nas):
#                 print(f"{i}-na: ", na)
#                 na_num = 0
#                 col = data.iloc[:, i]
#                 value = col.values.tolist()
#                 unique_value = list(set(value))
#                 unique_value_idx = [(u, (np.where(col == u)[0].tolist())[0]) for u in unique_value]
#                 uv = [(u, value.count(u)) for u in unique_value]
#                 print("unique_value: ", unique_value)
#                 print("unique_value_count: ", uv)
#                 print("unique_value_idx: ", unique_value_idx)
                
#                 na_value = []
#                 if type(na) == list:
#                     for n in na:
#                         if n in unique_value:
#                             unique_value.remove(n)
#                             na_num += value.count(n)
#                             na_value.append(n)
#                 else:
#                     if na in unique_value:
#                         unique_value.remove(na)
#                         na_num += value.count(na)
#                         na_value.append(na)
#                 print("after replace na-unique_value: ", unique_value)

#                 if len(unique_value) == 0:
#                     print(f"{key}'s column {col_names[i]} is empty")
#                     na_colnames.append(col_names[i])
#                     continue
#                 elif na_num > threshold*n_samples: 
#                     print(f"{key}'s column {col_names[i]} more than {threshold*100}% is empty")
#                     na_colnames.append(col_names[i])
#                     continue
#                 else:
#                     for j, n in enumerate(na_value):
#                         idxs = np.where(col == n)[0].tolist()
#                         radom_fill = random.choices(unique_value, k=len(idxs))
#                         data.iloc[idxs, i] = radom_fill
#             for na_colname in na_colnames:
#                 data.drop(na_colname, axis=1, inplace=True)

#             print("na_cols: ", len(na_colnames))
#             print("new data shape: ", data.shape)
#             print("original data shape: ", (n_samples, n_features))

#             save_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{dp_method}/{key}_{dp_budget}_{dp_number}.csv"
#             # save_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{key}.csv"
#             with open (save_path, 'w') as f:
#                 data.to_csv(f, index=False)

vm_target_cols = ['METAREAD', 'SPLIT', 'METRO', 'SCHOOL', 'SEX', 'RESPONDT', 'SLREC', 'LABFORCE', 'FARM', 'OWNERSHP', 'URBAN', 'SPANNAME', 'EMPSTAT', 'HISPAN', 'RACE', 'NCHLT5', 'GQ', 'MARST', 'SIZEPL', 'HISPRULE', 'SAMESEA5', 'SAMEPLAC', 'MIGTYPE5', 'MIGRATE5', 'MARRNO', 'INCNONWG', 'OWNERSHPD', 'FAMSIZE', 'EMPSTATD', 'MIGRATE5D', 'CLASSWKRD', 'EDUC', 'VETSTATD', 'HIGRADE', 'AGE', 'SUPDIST', 'COUNTY', 'URBPOP', 'RACED', 'SEA', 'HISPAND', 'ENUMDIST', 'BPL', 'MIGSEA5', 'OCC', 'HIGRADED', 'MIGPLAC5', 'IND', 'EDUCD', 'GQTYPED', 'MTONGUED', 'MBPLD', 'FBPLD', 'BPLD', 'INCWAGE']
vm_num_cols = ['NCHLT5', 'SIZEPL', 'FAMSIZE', 'EDUC', 'HIGRADE', 'AGE', 'SUPDIST', 'URBPOP', 'EDUCD']
vm_cat_cols = ['METAREAD', 'SPLIT', 'METRO', 'SCHOOL', 'SEX', 'RESPONDT', 'SLREC', 'LABFORCE', 'FARM', 'OWNERSHP', 'URBAN', 'SPANNAME', 'EMPSTAT', 'HISPAN', 'RACE', 'GQ', 'MARST', 'HISPRULE', 'SAMESEA5', 'SAMEPLAC', 'MIGTYPE5', 'MIGRATE5', 'MARRNO', 'INCNONWG', 'OWNERSHPD', 'EMPSTATD', 'MIGRATE5D', 'CLASSWKRD', 'VETSTATD', 'COUNTY', 'RACED', 'SEA', 'HISPAND', 'ENUMDIST', 'BPL', 'MIGSEA5', 'OCC', 'HIGRADED', 'MIGPLAC5', 'IND', 'GQTYPED', 'MTONGUED', 'MBPLD', 'FBPLD', 'BPLD', 'INCWAGE']

ar_target_cols = ['SPLIT', 'URBAN', 'SEX', 'RESPONDT', 'SLREC', 'LABFORCE', 'SCHOOL', 'FARM', 'OWNERSHP', 'SPANNAME', 'METRO', 'EMPSTAT', 'HISPAN', 'NCHLT5', 'RACE', 'MARST', 'GQ', 'SIZEPL', 'HISPRULE', 'MIGRATE5', 'SAMESEA5', 'SAMEPLAC', 'MIGTYPE5', 'INCNONWG', 'MARRNO', 'SEA', 'OWNERSHPD', 'FAMSIZE', 'EMPSTATD', 'MIGRATE5D', 'CLASSWKRD', 'HIGRADE', 'EDUC', 'VETSTATD', 'AGE', 'SUPDIST', 'COUNTY', 'HISPAND', 'URBPOP', 'RACED', 'BPL', 'MIGSEA5', 'OCC', 'IND', 'MIGPLAC5', 'GQTYPED', 'EDUCD', 'HIGRADED', 'ENUMDIST', 'METAREAD', 'MTONGUED', 'BPLD', 'FBPLD', 'MBPLD']
ar_num_cols = ['NCHLT5', 'SIZEPL', 'FAMSIZE', 'HIGRADE', 'EDUC', 'AGE', 'SUPDIST', 'URBPOP', 'EDUCD']
ar_cat_cols = ['SPLIT', 'URBAN', 'SEX', 'RESPONDT', 'SLREC', 'LABFORCE', 'SCHOOL', 'FARM', 'OWNERSHP', 'SPANNAME', 'METRO', 'EMPSTAT', 'HISPAN', 'RACE', 'MARST', 'GQ', 'HISPRULE', 'MIGRATE5', 'SAMESEA5', 'SAMEPLAC', 'MIGTYPE5', 'INCNONWG', 'MARRNO', 'SEA', 'OWNERSHPD', 'EMPSTATD', 'MIGRATE5D', 'CLASSWKRD', 'VETSTATD', 'COUNTY', 'HISPAND', 'RACED', 'BPL', 'MIGSEA5', 'OCC', 'IND', 'MIGPLAC5', 'GQTYPED', 'HIGRADED', 'ENUMDIST', 'METAREAD', 'MTONGUED', 'BPLD', 'FBPLD', 'MBPLD']

dp_methods = ["UCLANESL", "rmckenna", "DPSyn", "gardn999", "PrivBayes", "original_data"]
dp_budgets = ["0.3", "1.0", "8.0"]
dp_number = 1

for key in data_dict.keys():
    if key == "arizona":
        continue
    if key == "arizona":
        key_abbreviation = "az"
        target_cols = ar_target_cols
        cat_cols = ar_cat_cols
        num_cols = ar_num_cols
    elif key == "vermont":
        key_abbreviation = "vt"
        target_cols = vm_target_cols
        cat_cols = vm_cat_cols
        num_cols = vm_num_cols

    data_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{key}_nofill.csv"
    original_data = pd.read_csv(data_path, skipinitialspace=True)
    print(f"\n{key} original_data.shape", original_data.shape)
    print("original_data.head\n", original_data.head())
    valid_cat_col_values = {}
    for cat_col in cat_cols:
        original_col = original_data[cat_col]
        original_value = original_col.values.tolist()
        original_unique_value = list(set(original_value))
        valid_cat_col_values[cat_col] = original_unique_value

    for dp_method in dp_methods:
        for dp_budget in dp_budgets:
            nas = data_dict[key]
            print("value length: ", len(nas))

            if dp_method == "original_data" and dp_budget != "0.3":
                continue
            elif dp_method == "original_data" and dp_budget == "0.3":
                data = original_data.copy()
                save_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{dp_method}/{key}_nonorm.csv"
            else:
                data_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{dp_method}/out_{key_abbreviation}_{dp_budget}_{dp_number}.csv"
                data = pd.read_csv(data_path, skipinitialspace=True).astype(np.int64)
                save_path = f"/nfsdata/tianhao/dataset/Match 3/{key.capitalize()}/{dp_method}/{key}_{dp_budget}_{dp_number}_nonorm.csv"
            col_names = data.columns.values.tolist()
            n_samples, n_features = data.shape
            print("data shape: ", data.shape)
            print("data head: \n", data.head())
            print("col_names: \n", col_names)

            na_colnames = []
            threshold = 0.4
            for i, na in enumerate(nas):
                if col_names[i] not in target_cols:
                    na_colnames.append(col_names[i])
                    continue
                print(f"{i}-na: ", na)
                col_name = col_names[i]
                col = data.iloc[:, i]
                value = col.values.tolist()
                unique_value = list(set(value))
                unique_value_idx = [(u, (np.where(col == u)[0].tolist())[0]) for u in unique_value]
                uv = [(u, value.count(u)) for u in unique_value]
                print("unique_value: ", unique_value)
                # print("unique_value_count: ", uv)
                # print("unique_value_idx: ", unique_value_idx)
                
                if col_name not in valid_cat_col_values.keys():
                    na_value = []
                else:
                    na_value = list(set(unique_value) - set(valid_cat_col_values[col_name]))
                
                for n in na_value:
                    unique_value.remove(n)

                if type(na) == list:
                    for n in na:
                        if n in unique_value:
                            unique_value.remove(n)
                            na_value.append(n)
                else:
                    if na in unique_value:
                        unique_value.remove(na)
                        na_value.append(na)
                print("after replace na-unique_value: ", unique_value)

                if len(unique_value) == 0:
                    print(f"{key}'s column {col_names[i]} is empty")
                    na_colnames.append(col_names[i])
                    continue
                elif col_names[i] not in target_cols:
                    print(f"{key}'s column {col_names[i]} more than {threshold*100}% is empty")
                    na_colnames.append(col_names[i])
                    continue
                else:
                    for j, n in enumerate(na_value):
                        idxs = np.where(col == n)[0].tolist()
                        radom_fill = random.choices(unique_value, k=len(idxs))
                        data.iloc[idxs, i] = radom_fill
            for na_colname in na_colnames:
                data.drop(na_colname, axis=1, inplace=True)

            print("na_cols: ", len(na_colnames))
            print("new data shape: ", data.shape)
            print("original data shape: ", (n_samples, n_features))
            print("target_cols: ", len(target_cols))

            with open (save_path, 'w') as f:
                data.to_csv(f, index=False)


