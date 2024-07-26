import torch
import numpy as np
import os

from Properties import Properties
from ann.multi_label_ann import multi_label_ann


class Trainer:
    def __init__(self):

        self.dataset_name = None
        self.dataset = None
        self.parallel_out_put = []

        self.data_matrix = None
        self.label_matrix = None

        self.num_labels = 0
        self.num_instances = 0
        self.num_conditions = 0

    def initialization(self, para_dataset: Properties = None, para_data: np.ndarray = None,
                       para_label: np.ndarray = None):
        self.dataset = para_dataset
        self.parallel_out_put = []

        self.data_matrix = para_data
        self.label_matrix = para_label

        self.num_labels = self.label_matrix.shape[1]
        self.num_instances = self.data_matrix.shape[0]
        self.num_conditions = self.data_matrix.shape[1]

    def run_train(self):
        # parallel train
        train_parallel_feature = self.parallel_train()
        # full connect train
        self.full_train(train_parallel_feature)

        return self.parallel_out_put

    def parallel_train(self):

        flag = 0
        for index in range(self.num_labels):
            parallel_network = multi_label_ann(self.dataset.parallel_layer_num_nodes,
                                               self.dataset.learning_rate,
                                               self.dataset.parallel_activators,
                                               self.num_labels)
            parallel_network.bounded_train(100,
                                           10,
                                           0.0001,
                                           self.data_matrix,
                                           self.label_matrix,
                                           index)
            out_put = parallel_network.extract_features(self.data_matrix)

            if os.path.exists('../network/parallel_network/{}'.format(self.dataset.name)):
                torch.save(parallel_network.state_dict(),
                           '../network/parallel_network/{}/net{}.pkl'.format(self.dataset.name, index))
            else:
                os.makedirs('../network/parallel_network/{}'.format(self.dataset.name))
                torch.save(parallel_network.state_dict(),
                           '../network/parallel_network/{}/net{}.pkl'.format(self.dataset.name, index))
            # Collect features
            if flag == 0:
                out_put = out_put.cpu().detach().numpy()
                self.parallel_out_put = out_put
                flag = 1
            else:
                out_put = out_put.cpu().detach().numpy()
                self.parallel_out_put = np.hstack([self.parallel_out_put, out_put])
        return self.parallel_out_put

    def full_train(self, para_input):
        self.num_labels *= 2
        temp_Y = []
        for line in self.label_matrix:
            temp_line = []
            for v in line:
                if v == 1:
                    temp_line += [0, 1]
                else:
                    temp_line += [1, 0]
            temp_Y.append(temp_line)
        self.label_matrix = np.array(temp_Y)
        full_connect_network = multi_label_ann(self.dataset.full_connect_layer_num_nodes,
                                               self.dataset.learning_rate,
                                               self.dataset.full_connect_activators,
                                               self.num_labels)
        full_connect_network.bounded_train(1000,
                                           100,
                                           0.0001,
                                           para_input,
                                           self.label_matrix)

        # Save the second stage network locally
        if os.path.exists('../network/serial_network/{}'.format(self.dataset.name)):
            torch.save(full_connect_network.state_dict(),
                       '../network/serial_network/{}/net.pkl'.format(self.dataset.name))
        else:
            os.makedirs('../network/serial_network/{}'.format(self.dataset.name))
            torch.save(full_connect_network.state_dict(),
                       '../network/serial_network/{}/net.pkl'.format(self.dataset.name))
