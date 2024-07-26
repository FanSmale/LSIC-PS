import torch
import numpy as np

from Properties import Properties
from ann.multi_label_ann import multi_label_ann
from algorithm.Trainer import Trainer


def model_reader(net, device, model_name, save_src='./models/SimulateModel/'):
    print("Prepare to import the trained model")
    print("Read model: {}".format(save_src + model_name))
    model = torch.load(save_src + model_name)
    try:
        net.load_state_dict(model)
    except RuntimeError:
        print("The data is obtained by multi-card training, so re-read")
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in model.items():
            name = k[7:]
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)

    net = net.to(device)
    return net


class Tester(Trainer):
    def __init__(self):
        super().__init__()

        self.predicted_label = None
        self.actual_label = None
        self.metrics_value_dict = {}

    def initialization(self, para_dataset: Properties = None, para_data: np.ndarray = None,
                       para_label: np.ndarray = None, para_parallel_out_put=None):

        if para_parallel_out_put is None:
            para_parallel_out_put = []
        self.dataset = para_dataset
        self.parallel_out_put = []

        self.data_matrix = para_data
        self.label_matrix = para_label

        self.num_labels = self.label_matrix.shape[1]
        self.num_instances = self.data_matrix.shape[0]
        self.num_conditions = self.data_matrix.shape[1]

        self.parallel_out_put = para_parallel_out_put

        self.predicted_label = None
        self.actual_label = None

    def run_test(self):

        test_parallel_feature = self.parallel_test()
        # predicted matrix
        self.predicted_label = self.full_test(test_parallel_feature)
        # original matrix
        self.actual_label = self.label_matrix

    def parallel_test(self):

        out_test_put_matrix = []
        flag = 0
        for index in range(self.num_labels):
            test_parallel_network = multi_label_ann(self.dataset.parallel_layer_num_nodes,
                                                    self.dataset.learning_rate,
                                                    self.dataset.parallel_activators,
                                                    self.num_labels)
            test_parallel_network = model_reader(test_parallel_network,
                                                 torch.device('cuda'),
                                                 "net{}.pkl".format(index),
                                                 save_src='../network/parallel_network/{}/'.format(
                                                     self.dataset.name))

            out_test_put = test_parallel_network.extract_features(self.data_matrix)
            if flag == 0:
                out_test_put = out_test_put.cpu().detach().numpy()
                out_test_put_matrix = out_test_put
                flag = 1
            else:
                out_test_put = out_test_put.cpu().detach().numpy()
                out_test_put_matrix = np.hstack([out_test_put_matrix, out_test_put])
        return out_test_put_matrix

    def full_test(self, para_input):
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

        test_full_connect_network = multi_label_ann(self.dataset.full_connect_layer_num_nodes,
                                                    self.dataset.learning_rate,
                                                    self.dataset.full_connect_activators,
                                                    self.num_labels)
        test_full_connect_network = model_reader(test_full_connect_network, torch.device('cuda'),
                                                 "net.pkl",
                                                 save_src='../network/serial_network/{}/'.format(
                                                     self.dataset.name))
        # Get the final prediction results
        test_full_connect_network.predict(para_input, self.label_matrix)
        predicted_label = test_full_connect_network.prediction_tensor
        predicted_label = predicted_label.cpu().detach().numpy()
        return predicted_label
