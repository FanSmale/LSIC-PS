import numpy as np
import scipy.io as sio

from sklearn.model_selection import KFold
from Properties import Properties

from algorithm.Trainer import Trainer
from algorithm.Tester import Tester


def read_data(para_train_filename: str = ""):
    # Read mat data.

    temp_all_data_read = sio.loadmat(para_train_filename)
    temp_train_data_read = np.array(temp_all_data_read['train_data'])  # num_instances * num_conditions
    temp_test_data_read = np.array(temp_all_data_read['test_data'])
    temp_train_targets_read = np.array(temp_all_data_read['train_target']).transpose()
    temp_test_targets_read = np.array(temp_all_data_read['test_target']).transpose()
    temp_train_targets_read[np.where(temp_train_targets_read < 0.0)] = 0.0
    temp_test_targets_read[np.where(temp_test_targets_read < 0.0)] = 0.0
    temp_train_data_read = (temp_train_data_read - temp_train_data_read.min(axis=0)) / \
                           (temp_train_data_read.max(axis=0) - temp_train_data_read.min(axis=0) + 0.0001)
    temp_test_data_read = (temp_test_data_read - temp_test_data_read.min(axis=0)) / \
                          (temp_test_data_read.max(axis=0) - temp_test_data_read.min(axis=0) + 0.0001)

    temp_sum = np.sum(temp_train_targets_read)
    temp_area_in_train = temp_train_targets_read.size
    temp_ones_in_train = (temp_sum + temp_area_in_train) / 2
    temp_proportion = temp_ones_in_train / temp_area_in_train

    print("Proportion of 1 in train target (label matrix): ", temp_ones_in_train, " out of ", temp_area_in_train,
          " gets ", temp_proportion)

    return temp_train_data_read, temp_train_targets_read, temp_test_data_read, temp_test_targets_read


def for_kf(kf_num: int = 5):
    dataset = Properties("Birds")

    train_data_matrix, train_label_matrix, test_data_matrix, test_label_matrix = read_data(dataset.filename)
    data_matrix = np.vstack((train_data_matrix, test_data_matrix))
    label_matrix = np.vstack((train_label_matrix, test_label_matrix))

    tr = Trainer()
    te = Tester()
    metrics_dict = {"Peak F1-Score": [], "NDCG": [], "Macro-AUC": [], "Micro-AUC": [],
                    "Coverage": [], "One Error": [], "Ranking Loss": [], "Hamming Loss": []}

    kf = KFold(kf_num, shuffle=True)
    for k, (train_index, test_index) in enumerate(kf.split(data_matrix)):
        train_data_matrix = data_matrix[train_index, :]
        test_data_matrix = data_matrix[test_index, :]
        train_label_matrix = label_matrix[train_index, :]
        test_label_matrix = label_matrix[test_index, :]

        ############
        # training #
        ############
        tr.initialization(dataset, train_data_matrix, train_label_matrix)
        parallel_out_put = tr.run_train()
        ###########
        # testing #
        ###########
        te.initialization(dataset, test_data_matrix, test_label_matrix,
                          para_parallel_out_put=parallel_out_put)
        te.run_test()



if __name__ == "__main__":
    for_kf()
