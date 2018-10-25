from __future__ import division
from collections import OrderedDict
from commons import Variable
import numpy as np

dtype = np.float32


class DenseLayer(object):
    """A fully connected dense layer

    Parameters:
        w: in_dim x out_dim: The weight matrix
        n: out_dim, : the bias

    Arguments:
        n_out: number of output features
        init_type: the type of initialization
            can be gaussian or uniform

    """

    def __init__(self, n_out, init_type):
        self.n_out = n_out
        self.init_type = init_type
        self.params = OrderedDict()
        self.params["w"] = Variable()
        self.params["b"] = Variable()

    def get_output_dim(self):
        # The output dimension
        return self.n_out

    def init(self, n_in):
        # initializing the network, given input dimension
        scale = np.sqrt(1. / (n_in))
        if self.init_type == "gaussian":
            self.params["w"].value = scale * np.random.normal(
                0, 1, (n_in, self.n_out)).astype(dtype)
        elif self.init_type == "uniform":
            self.params["w"].value = 2 * scale * np.random.rand(
                n_in, self.n_out).astype(dtype) - scale
        else:
            raise NotImplementedError("{0} init type not found".format(
                self.init_type))
        self.params["b"].value = np.zeros((self.n_out), dtype=dtype)

    def forward(self, inputs):
        """The forward pass

        Arguments:
            inputs (``OrderedDict``): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array n_in dimensions

        Returns:
            outputs (``OrderedDict``): A dictionary containing
                height: The height of the output (1 for a dense layer)
                width: The width of the output (1 for a dense layer)
                out_channels: The output number of feature maps
                    Same as the input channels for this layer
                data: a flattened output data array of the form
                    n_out dimensions

        """
        data = inputs["data"]
        outputs = OrderedDict()
        # cache for backward pass
        self.data = data
        outputs["height"] = 1
        outputs["width"] = 1
        outputs["channels"] = self.n_out
        # [batch_size, n_in] * [n_in, n_out] + [n_out] = [batch_size, n_out]
        outputs["data"] = np.dot(data, self.params["w"].value) + self.params["b"].value
        return outputs

    def backward(self, output_grads):
        """The backward pass

        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that this layer also computes the gradients wrt the
        parameters (i.e you should populate the values of
        self.params["w"].grad, and self.params["b"].grad here)

        Note that you should compute the average gradient
        (i.e divide by batch_size) when you computing the gradient
        of parameters.
        """
        # [batch_size, n_out]
        grad = output_grads["grad"]
        batch_size = grad.shape[0]
        self.params["w"].grad = np.zeros_like(self.params["w"].value, dtype=dtype)
        for i in range(batch_size):
            # [n_in, n_out] = outer([n_in], [n_out])
            self.params["w"].grad += np.outer(self.data[i], grad[i])
        self.params["w"].grad = self.params["w"].grad / dtype(batch_size)

        self.params["b"].grad = np.zeros_like(self.params["b"].value, dtype=dtype)
        for i in range(batch_size):
            self.params["b"].grad += grad[i]
        self.params["b"].grad = self.params["b"].grad / dtype(batch_size)
        input_grads = OrderedDict()
        # [batch_size, n_in] = [batch_size, n_out] * [n_out, n_in]
        input_grads["grad"] = np.dot(grad, self.params["w"].value.transpose())
        assert input_grads["grad"].shape == self.data.shape
        return input_grads


class ReLULayer(object):
    """A ReLU activation layer
    """

    def __init__(self):
        self.params = OrderedDict()

    def forward(self, inputs):
        """The forward pass

        Arguments:
            inputs (``OrderedDict``): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array n_in dimensions

        Returns:
            outputs (``OrderedDict``): A dictionary containing
                height: The height of the output (1 for a dense layer)
                width: The width of the output (1 for a dense layer)
                out_channels: The output number of feature maps
                    Same as the input channels for this layer
                data: a flattened output data array of the form
                    n_in

        Note that you only need to populate the outputs["data"]
        element.
        """
        outputs = OrderedDict()
        for key in inputs:
            if key != "data":
                outputs[key] = inputs[key]
            else:
                # hash for backward pass
                self.data = inputs[key]
                output_data = np.copy(self.data)
                output_data[output_data < 0] = 0
                outputs["data"] = output_data
        return outputs

    def backward(self, outputs_grad):
        """The backward pass

        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that you just compute the gradient wrt the ReLU layer
        """
        input_grads = OrderedDict()
        grad = outputs_grad["grad"]
        grad[self.data < 0] = 0
        input_grads["grad"] = grad
        assert input_grads["grad"].shape == self.data.shape
        return input_grads
