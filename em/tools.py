import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, contingency_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def detection_outlier(dfcolumn):
    """
    This function is used to detect outliers for each numerical attribute
    Input:
        dfcolumn: numerical attribute
    Output:
        None
    """
    #Calculate Q1, Q3 and IQR
    Q1 = dfcolumn.quantile(0.25)
    Q3 = dfcolumn.quantile(0.75)
    IQR = Q3-Q1 #Interquartile range
    #In general, constant should be 1.
    lower  = Q1-1*IQR
    higher = Q3+1*IQR
    #Find number of outliers for specific column
    print('Before data preprocessing:')
    print('Skewness:',dfcolumn.skew())
    print(dfcolumn.describe())
    dfcolumn.loc[(dfcolumn> higher) | (dfcolumn< lower)]=dfcolumn.mean()
    print('After replacing outliers by mean:')
    print('Skewness:',dfcolumn.skew())
    print('Median:',dfcolumn.median())
    print('IQR value:',IQR)
    print('Lower,Higher:',lower,',',higher)
    return 


def calculate_purity(y_true, y_pred):
    """
    This function is used to calculate purity
    Input:
        y_true: true labels
        y_pred: predicted labels
    Output:
        purity: purity
    """
    conti_matrix = contingency_matrix(y_true, y_pred)
    purity = np.sum(np.amax(conti_matrix, axis=0)) / np.sum(conti_matrix)
    return purity


def create_directory(path):
    """
    This function is used to create a directory
    Input:
        path: directory path
    Output:
        None
    """
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory '{path}': {error}")


def pred2prob(y_pred):
    """
    This function is used to convert prediction to probability
    Input:
        y_pred: prediction
    Output:
        y_prob: probability
    """
    num_labels = len(y_pred)
    num_classes = max(y_pred) + 1
    y_prob = np.zeros((num_labels, num_classes))
    for i in range(num_labels):
        y_prob[i, y_pred[i]] = 1
    return y_prob


def output_metrics(y_pred, y_prob, data, dict_centers):
# def output_metrics(y_pred, y_prob, data):
    """
    This function is used to output metrics
    Input:
        y_pred: predicted labels
        y_prob: probability
        data: data
        dict_centers: dictionary of centers
    Output:
        mse: mean squared error
        mae: mean absolute error
    """
    data = np.array(data)

    # #Calculate mse and mae in batches
    # batch_size = 15000
    # total_rows = len(labels)
    # for start_idx in range(0, total_rows, batch_size):
    #     end_idx = min(start_idx + batch_size, total_rows)
    #     print("batch", start_idx, "--", end_idx, "/", total_rows)
    #     data_batch = data[start_idx:end_idx]
    #     mse_distances = np.sum((data_batch[:, np.newaxis] - centers)**2, axis=2)
    #     mae_distances = np.sum(np.abs(data_batch[:, np.newaxis] - centers), axis=2)
    #     min_mse_distances = np.min(mse_distances, axis=1)
    #     min_mae_distances = np.min(mae_distances, axis=1)
    #     mse += np.sum(min_mse_distances)
    #     mae += np.sum(min_mae_distances)

    #Calculate mse and mae directly
    if dict_centers['empty'] != []:
        for label in dict_centers['empty']:
            y_prob[:, label] = 0

    centers = np.array(dict_centers['centers'])
    l2_distances = np.sum((data[:, np.newaxis] - centers)**2, axis=2)
    l1_distances = np.sum(np.abs(data[:, np.newaxis] - centers), axis=2)
    normalized_l2_distances = l2_distances*y_prob
    normalized_l1_distances = l1_distances*y_prob
    mse = np.sum(normalized_l2_distances)/data.shape[0]
    mae = np.sum(normalized_l1_distances)/data.shape[0]

    return mse, mae
    # #calculate ari...
    # ari = adjusted_rand_score(labels, y_pred)
    # nmi = normalized_mutual_info_score(labels, y_pred)
    # p = calculate_purity(labels, y_pred)
    # silhouette_avg = silhouette_score(data, y_pred)
    # calinski_harabasz_avg = calinski_harabasz_score(data, y_pred)
    # davies_bouldin_avg = davies_bouldin_score(data, y_pred)
    # return mse, mae, ari, nmi, p, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg


# def output_metrics_prob(y_pred, data, y_prob):
# # def output_metrics_prob(labels, y_pred, data, y_prob):
#     """
#     This function is used to output metrics
#     Input:
#         labels: true labels
#         y_pred: predicted labels
#         data: data
#         y_prob: probability
#     Output:
#         mse: mean squared error
#         mae: mean absolute error
#         ari: adjusted rand index
#         nmi: normalized mutual information
#         p: purity
#         silhouette_avg: silhouette score
#         calinski_harabasz_avg: calinski harabasz score
#         davies_bouldin_avg: davies bouldin score
#     """

#     labels = np.array(y_pred).reshape(-1,)
#     data = np.array(data)

#     # # Calculate mse and mae in batches
#     # mse = 0
#     # mae = 0
#     # batch_size = 15000
#     # total_rows = len(labels)
#     # for start_idx in range(0, total_rows, batch_size):
#     #     end_idx = min(start_idx + batch_size, total_rows)
#     #     print("batch", start_idx, "--", end_idx, "/", total_rows)
#     #     data_batch = data[start_idx:end_idx]
#     #     prob_batch = y_prob[start_idx:end_idx]
#     #     mse_distances = np.sum((data_batch[:, np.newaxis] - centers)**2, axis=2)
#     #     mae_distances = np.sum(np.abs(data_batch[:, np.newaxis] - centers), axis=2)
#     #     mse += np.sum(mse_distances*prob_batch)
#     #     mae += np.sum(mae_distances*prob_batch)
#     # mse /= total_rows
#     # mae /= total_rows

#     #Calculate mse and mae directly

#     if centers.shape[0] != np.unique(labels).shape[0]:
#         print("Warning: labels and centers do not match!")
#         mse = None
#         mae = None
#     else:
#         l2_distances = np.sum((data[:, np.newaxis] - centers)**2, axis=2)
#         l1_distances = np.sum(np.abs(data[:, np.newaxis] - centers), axis=2)
#         total_weights = np.sum(y_prob, axis=0)
#         normalized_l2_distances = l2_distances*y_prob
#         normalized_l1_distances = l1_distances*y_prob
#         mse = np.sum(normalized_l2_distances/total_weights)
#         mae = np.sum(normalized_l1_distances/total_weights)

#     mse = 0
#     mae = 0
#     unique_labels = np.unique(labels)
#     total_rows = 0
#     for label in unique_labels:
#         cluster_idx = np.where(labels == label)[0]
#         cluster_data = data[cluster_idx]
#         centroid = np.mean(cluster_data, axis=0)

#         prob_idx = np.where(p[:, label-1])[0]
#         class_data = data[prob_idx]
#         probability = y_prob[prob_idx, label-1]
#         # print("class_data.shape", class_data.shape)
#         # print("centroid.shape", centroid.shape)
#         l2_distances = np.sum((class_data - centroid)**2, axis=1)
#         # print("l2_distances.shape", l2_distances.shape)
#         # print("probability.shape", probability.shape)
#         l1_distances = np.sum(np.abs(class_data - centroid), axis=1)
        
#         mse += np.sum(l2_distances*probability)
#         # print("mes", mse)
#         mae += np.sum(l1_distances*probability)
#         total_rows += np.sum(probability)
#     mse /= total_rows
#     mae /= total_rows

#     return mse, mae
#     # how to utilize probability to calculate ari...?
#     # ari = adjusted_rand_score(labels, y_pred)
#     # nmi = normalized_mutual_info_score(labels, y_pred)
#     # p = calculate_purity(labels, y_pred)
#     # silhouette_avg = silhouette_score(data, y_pred)
#     # calinski_harabasz_avg = calinski_harabasz_score(data, y_pred)
#     # davies_bouldin_avg = davies_bouldin_score(data, y_pred)

#     # return mse, mae, ari, nmi, p, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg


# def find_closest_point(class_data, label, centroid):
#     """
#     This function is used to find the closest point to the centroid
#     Input:
#         class_data: data of the class
#         label: label of the class
#         centroid: centroid of the class
#     Output:
#         closest_point: closest point to the centroid
#     """
#     distances = cdist(class_data, np.array([centroid]))
#     closest_index = class_indices[np.argmin(distances)]
#     return class_data[closest_index]


def calculate_centers(data, labels):
    """
    This function is used to calculate centers
    Input:
        data: data
        labels: labels
    Output:
        class_centroids: class centroids
    """
    num_classes = max(labels) + 1
    # num_classes = np.unique(labels).shape[0]
    print("num_classes", num_classes)
    class_centroids = []
    empty_classes = []

    for _, label in enumerate(range(num_classes)):
        class_indices = np.where(labels == label)[0]
        # indices = (labels == label).nonzero()[0]
        if len(class_indices) == 0:
            class_centroids.append(np.array([-1]*data.shape[1]))
            empty_classes.append(label)
            continue
        # print("class_indices", class_indices)
        class_data = data[class_indices]
        centroid = np.mean(class_data, axis=0)
        class_centroids.append(centroid)

    print("Successfully calculate centers", len(class_centroids))
    dict_centers = {
        "centers": class_centroids,
        "empty": empty_classes
    }
    return dict_centers


# def calculate_centers_prob(data, y_prob):
#     """
#     This function is used to calculate centers
#     Input:
#         data: data
#         y_prob: probability
#     Output:
#         class_centroids: class centroids
#     """
#     class_centroids = np.dot(y_prob.T, data)/prob.shape[0]
#     print("data.shape", data.shape)
#     print("y_prob.shape", y_prob.shape)
#     print("class_centroids.shape", class_centroids.shape)
#     return class_centroids