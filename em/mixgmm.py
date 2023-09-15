# GaussianMixture Clustering
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import os
from tools import create_directory


# 1. Read data
print("Start reading data")
domain = "UCLAdult"
kfreq = domain+"_kfreq"
name = domain+"_mix"
input_name = domain+"_sparse_features"
input_path = "./dataset/" + domain + "/" + input_name + ".csv"
data = pd.read_csv(input_path, skipinitialspace=True, header=None)
print("data.shape", data.shape)


# 2. Set parameters
num_classes = 400
K = list(range(2, 11))+[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, \
                        130, 140, 150, 160, 170, 180, 190, 200]

# covariance_types = ['full', 'tied', 'diag', 'spherical']
covariance_types = ['tied']
n_init = 1

output_dir = "./result/"+name+"/prediction/"
prediction_dir = output_dir+"y_pred/"
probability_dir = output_dir+"y_prob/"
if not os.path.exists(prediction_dir):
    create_directory(prediction_dir)
if not os.path.exists(probability_dir):
    create_directory(probability_dir)

exist_dir = output_dir+"exist/"
exist_prediction_dir = exist_dir+"y_pred/"
exist_probability_dir = exist_dir+"y_prob/"
if not os.path.exists(exist_prediction_dir):
    create_directory(exist_prediction_dir)
if not os.path.exists(exist_probability_dir):
    create_directory(exist_probability_dir)


# 3. Apply GaussianMixture
print("Start applying GaussianMixture")
begin_time = time.time()
kfreq_dir = "./famd/result/"+kfreq+"/prediction/y_pred/"
for covariance_type in covariance_types:
    result = {}
    output_path = output_dir+f"{covariance_type}1.csv"
    with open(output_path, 'w') as f:
        f.write("classes,k,n_components,covariance_type,time,prediction_path,probability_path\n")

    for idx, k in enumerate(K):
        # get kfreq labels
        kfreq_path = kfreq_dir+domain+f"_sparse_{k}.csv"
        kfreq_labels = pd.read_csv(kfreq_path, skipinitialspace=True, header=None).astype(np.int32)
        max_component = int(num_classes/k)
        if max_component > 10:
            n_components = list(range(1, 10))+list(range(10, max_component, 10))+[max_component]
        else:
            n_components = list(range(1, max_component+1))

        for n_component in n_components:
            # initialize prediction
            offset = 0
            predict_result = np.zeros((data.shape[0], 1))

            # apply GMM
            for label in range(k):
                print(f"Processing label {label+1}/{k}...")
                # get data batch
                indices = np.where(kfreq_labels == label)[0]
                data_batch = data.iloc[indices]
                data_batch = np.array(data_batch)
                n_batch = min(n_component, data_batch.shape[0])
                print(f"Trying n_components={n_batch}, covariance_type={covariance_type}, n_init={n_init}")
                
                start_time = time.time()
                # GMM
                model = GaussianMixture(n_components=n_batch, \
                                        n_init=n_init, \
                                        covariance_type=covariance_type, \
                                        random_state=42
                                        )
                model.fit(data)
                y_pred = model.predict(data_batch)
                y_pred += offset
                y_pred = y_pred.reshape(-1, 1)
                predict_result[indices] = y_pred
                offset += n_batch
                #print time
                elapsed_time = time.time() - start_time
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
            
            #save result
            predict_set = set(predict_result.reshape(-1))
            classes = len(predict_set)
            if classes not in result.keys():
                result[classes] = {}
                result[classes]['k'] = k
                result[classes]['n_components'] = n_component
                result[classes]['time'] = elapsed_time

                prediction_path = prediction_dir+f"{covariance_type}_{classes}.csv"
                probability_path = probability_dir+f"{covariance_type}_{classes}.csv"
                result[classes]['prediction_path'] = prediction_path
                result[classes]['probability_path'] = probability_path
            else:
                print(f"classes={classes} already exists, k={k}, n_components={n_component}")
                prediction_path = exist_prediction_dir+f"{covariance_type}_{classes}_{k}_{n_component}.csv"
                probability_path = exist_probability_dir+f"{covariance_type}_{classes}_{k}_{n_component}.csv"
            np.savetxt(prediction_path, predict_result, delimiter=",", fmt="%d")
            np.savetxt(probability_path, model.predict_proba(data), delimiter=",", fmt="%s")

    #write result
    with open(output_path, 'a') as f:
        print(f"Writing result to {covariance_type}.csv")
        sorted_keys = sorted(result.keys())
        for classes in sorted_keys:
            f.write(f"{classes},{result[classes]['k']},{result[classes]['n_components']},{covariance_type},{result[classes]['time']},{result[classes]['prediction_path']},{result[classes]['probability_path']}\n")

end_time = time.time() - begin_time
print(f"Total time elapsed: {end_time:.2f} seconds")
