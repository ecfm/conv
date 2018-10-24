from __future__ import division
from layers import DataLayer, ConvLayer, PoolLayer,\
    DenseLayer, ReLULayer, LossLayer
from collections import OrderedDict
import numpy as np
import pdb


class LeNet(object):
    """The LeNet Class
    """

    def __init__(self, layers):
        self._network = []
        for layer in layers:
            layer_type = layer.pop("type")
            if layer_type == "data":
                # this is a data layer
                new_layer = DataLayer(**layer)
            elif layer_type == "conv":
                new_layer = ConvLayer(**layer)
            elif layer_type == "pool":
                new_layer = PoolLayer(**layer)
            elif layer_type == "dense":
                new_layer = DenseLayer(**layer)
            elif layer_type == "relu":
                new_layer = ReLULayer()
            elif layer_type == "loss":
                new_layer = LossLayer(**layer)
            else:
                raise NotImplementedError(
                    "Layer type: {0} not found".format(layer_type))
            self._network.append(new_layer)
        self.initialize()

    def initialize(self):
        data_layer = self._network[0]
        h = data_layer.height
        w = data_layer.width
        c = data_layer.channel
        for layer in self._network[1:]:
            if isinstance(layer, PoolLayer):
                layer.init(h, w, c)
                h, w, c = layer.get_output_dim(
                    h, w, layer.pad, layer.stride, layer.kernel_size, c)
            elif isinstance(layer, ConvLayer):
                layer.init(h, w, c)
                h, w, c = layer.get_output_dim(
                    h, w, layer.pad, layer.stride,
                    layer.kernel_size, layer.n_out_channels
                )
            elif isinstance(layer, DenseLayer):
                layer.init(h * w * c)
                h, w = 1, 1
                c = layer.get_output_dim()

    def forward(self, batch_x, batch_y):
        """The forward pass

        Arguments:
            batch_x (np.ndarray): num_samples x num_features (784):
                The training examples
            batch_y (np.ndarray): num_samples x 1: The training labels

        Returns:
            cost (OrderedDict): An OrderedDict containing
                cp: The loss value
                percent: The accuracy
                grad: The gradient for the loss layer
        """
        inputs = OrderedDict({"data": batch_x})
        for ix in range(0, len(self._network) - 1):
            inputs = self._network[ix].forward(inputs)

        # now compute the loss
        cost = self._network[-1].forward(
            inputs, return_grad=True, gold_labels=batch_y)
        assert "grad" in cost, "Grad not found"
        return cost

    def backward(self, cost):
        """The backward pass

        Arguments:
            cost (OrderedDict): An OrderedDict containing
                cp: the loss value
                percent: The accuracy
                grad: The gradient
        """
        back_grad = cost
        for ix in range(len(self._network) - 2, 0, -1):
            back_grad = self._network[ix].backward(back_grad)

    def parameters(self):
        """
        Iterates over the layers, and returns an ``OrderedDict``
        with parameters

        Returns:
            param_groups (``OrderedDict``): An OrderedDict mapping layer
                index to its parameters
        """
        # We ignore the Data Layer and the Loss Layer
        param_groups = OrderedDict()
        for ix in range(1, len(self._network) - 1):
            param_groups[ix] = self._network[ix].params
        return param_groups

    def save_model(self, filename):
        '''Save parameters to a file'''
        params = self.parameters()
        param_list = []
        for group in params:
            for w in params[group]:
                param_list.append(params[group][w].value)
        np.save(open(filename, "wb"), param_list)

    def load_model(self, filename):
        '''Take in a filename and load it'''
        param_list = np.load(filename)
        params = self.parameters()
        index = 0
        for group in params:
            for w in params[group]:
                assert params[group][w].value.shape == param_list[index].shape
                params[group][w].value = param_list[index]
                index += 1


def get_lenet_layers():
    """A dictionary containing the LeNet Parameters
    """
    layers = [OrderedDict() for _ in range(9)]

    # The first layer is a DATA layer
    layers[0] = OrderedDict()
    layers[0]["type"] = "data"
    # Define the input shape
    layers[0]["height"] = 28
    layers[0]["width"] = 28
    layers[0]["channel"] = 1

    # The second layer is a conv layer
    layers[1]["type"] = 'conv'
    layers[1]["n_out_channels"] = 20  # number of output channel
    layers[1]["kernel_size"] = 5  # kernel size
    layers[1]["stride"] = 1  # stride size
    layers[1]["pad"] = 0  # padding size
    layers[1]["group"] = 1  # group of input feature maps. You can ignore this

    # The third layer is a maxpooling layer
    layers[2]["type"] = 'pool'  # third layer is pooling layer
    layers[2]["act_type"] = 'max'  # use maxpooling
    layers[2]["kernel_size"] = 2  # kernel size
    layers[2]["stride"] = 2  # stride size
    layers[2]["pad"] = 0  # padding size

    # The fourth layer is a conv layer
    layers[3]["type"] = 'conv'
    layers[3]["kernel_size"] = 5
    layers[3]["stride"] = 1
    layers[3]["pad"] = 0
    layers[3]["group"] = 1
    layers[3]["n_out_channels"] = 50

    # The fifth layer is a maxpool
    layers[4]["type"] = 'pool'
    layers[4]["act_type"] = 'max'
    layers[4]["kernel_size"] = 2
    layers[4]["stride"] = 2
    layers[4]["pad"] = 0

    # The sixth layer is an inner product layer
    layers[5]["type"] = 'dense'  # inner product layer
    layers[5]["n_out"] = 500  # number of output dimensions
    layers[5]["init_type"] = 'uniform'  # initialization method

    # The seventh is a ReLU layer
    layers[6]["type"] = "relu"  # ReLU layer

    # The eight layer is a dense layer
    # projecting to the logits space
    layers[7]["type"] = "dense"  # logits layer
    layers[7]["n_out"] = 10
    layers[7]["init_type"] = "uniform"

    # The final layer is the loss layer
    layers[8]["type"] = "loss"  # loss layer
    layers[8]["n_classes"] = 10  # number of classes (10)

    return layers
