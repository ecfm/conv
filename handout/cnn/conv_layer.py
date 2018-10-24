from __future__ import division
import numpy as np
from collections import OrderedDict
from commons import Variable


dtype = np.float32


class BaseConvLayer(object):
    """The base convolution class

    Both the Convolutional layer and maxpool layer
    inherit from this.
    This also provides access to some commonly used functions
    like ``col2im_conv`` and ``im2col_conv``. See the pdf for
    information on both.

    """

    def __init__(self, **kwargs):
        assert hasattr(self, "kernel_size"), "Need to fix kernel size"
        assert hasattr(self, "stride"), "Need to fix kernel size"

    def col2im_conv(self, col, h_in, w_in, c, h_out, w_out):
        """Converts a flattened array into a input image structure

        Arguments:
            col: kernel_size * kernel_size * c_in, h_out * w_out
            h_in : input image height
            w_in : input image width
            c_in : input channels
            h_out : output_height
            w_out : output_width
        Returns:
            input_n : h_in x w_in x c_in
        """
        k = self.kernel_size
        pad = self.pad
        stride = self.stride
        im = np.zeros((h_in, w_in, c), dtype=dtype)
        col = col.flatten().reshape((k * k * c, h_out * w_out))
        for h in range(h_out):
            for w in range(w_out):
                left_r = h * stride
                right_r = (h * stride) + k
                left_c = w * stride
                right_c = w * stride + k
                curr_kernel = col[:, w + (h * w_out)]
                im[left_r: right_r, left_c: right_c, :] += \
                    curr_kernel.reshape((k, k, c))
        im = im[pad: im.shape[0] - pad, pad: im.shape[1] - pad, :]
        return im

    def im2col_conv(self, input_n, h_in, w_in, c_in, h_out, w_out):
        """Unrolls the kernels in a row major format

        Arguments:
            input_n : 1 x h_in * w_in * c_in
            h_in : input image height
            w_in : input image width
            c_in : input channels
            h_out : output_height
            w_out : output_width
        Returns:
            col : kernel_size * kernel_size * c_in, h_out * w_out
        """
        im = input_n.flatten().reshape((h_in, w_in, c_in))
        k = self.kernel_size
        stride = self.stride
        col = np.zeros((k * k * c_in, h_out * w_out), dtype=dtype)
        for h in range(h_out):
            for w in range(w_out):
                matrix_hw = im[(h * stride): (h * stride) + k, (w * stride): (w * stride) + k]
                col[:, w + h * w_out] = matrix_hw.flatten()
        return col

    def get_output_dim(self, h_in, w_in, pad, stride, k, out_channels):
        """Gets the output shape given input shapes, padding, stride
        and channels
        """
        h_out = ((h_in + (2 * pad) - k) // stride) + 1
        w_out = ((w_in + (2 * pad) - k) // stride) + 1
        return h_out, w_out, out_channels


class ConvLayer(BaseConvLayer):
    """The basic convolutional layer. Note that
    this actually performs cross correlation
    instead of convolutions.

    Arguments:
        n_out_channels: The number of out channels
        kernel_size: The size of the kernel
        stride: The stride of the kernel
        pad: Padding
        group: Group for tying weights. (We use 1)
    """

    def __init__(self, n_out_channels, kernel_size,
                 stride, pad, group):
        self.n_out_channels = n_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.group = group

        # Parameters
        self.params = OrderedDict()
        self.params["w"] = Variable()
        self.params["b"] = Variable()
        super(ConvLayer, self).__init__()

    def init(self, height, width, in_channels):
        """Initializing the conv parameters

        Parameters:
            w: kernel * kernel * in_channel x out_channel:
                the weight matrix for convolutions, flattened
            b: out_channel: the bias of the kernels
        """
        scale = np.sqrt(1. / (self.kernel_size * self.kernel_size * in_channels))
        in_dim = self.kernel_size * self.kernel_size * in_channels / self.group
        out_dim = self.n_out_channels
        self.params["w"].value = (2. * scale * np.random.rand(int(in_dim), int(out_dim)).astype(dtype)) - scale
        self.params["b"].value = np.zeros((1, out_dim), dtype=dtype)

    def forward(self, inputs):
        """The forward pass of the conv layer

        Arguments:
            inputs (OrderedDict): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array of the form
                    height * width * channel, unrolled in
                    the same way
        Returns:
            outputs (OrderedDict): A dictionary containing
                height: The height of the output
                width: The width of the output
                out_channels: The output number of feature maps
                data: a flattened output data array of the form
                    height * width * channel, unrolled in
                    the same way
        """
        h_in = inputs["height"]
        w_in = inputs["width"]
        c = inputs["channels"]
        batch_size = inputs["data"].shape[0]
        data = inputs["data"]

        # Save for backward pass
        self.h_in, self.w_in, self.c = h_in, w_in, c
        self.data = data

        k = self.kernel_size
        group = self.group
        h_out, w_out, num = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, self.n_out_channels
        )

        # Resolve output shape
        outputs = OrderedDict()
        outputs["height"] = h_out

        outputs["width"] = w_out

        outputs["channels"] = num

        outputs["data"] = np.zeros(
            (batch_size, h_out * w_out * num), dtype=dtype)

        for ix in range(batch_size):
            col = self.im2col_conv(data[ix], h_in, w_in, c, h_out, w_out)
            tmp_outputs = np.zeros((h_out * w_out, num), dtype=dtype)
            for g in range(group):
                prev_g = g * k * k * c // group
                next_g = (g + 1) * k * k * c // group
                col_g = col[prev_g: next_g, :]
                left_w = g * num // group
                right_w = (g + 1) * num // group
                weight = self.params["w"].value[:, left_w: right_w]
                b = self.params["b"].value[:, left_w: right_w]
                tmpval = np.dot(col_g.transpose(), weight) + b
                left_num = (g * num // group)
                right_num = ((g + 1) * num // group)
                tmp_outputs[:, left_num: right_num] = tmpval
            outputs["data"][ix] = tmp_outputs.flatten()
        return outputs

    def backward(self, output_grads):
        """
        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that we compute the output heights, widths, and
        channels on the fly in the backward pass as well.

        """
        h_in, w_in, c = self.h_in, self.w_in, self.c
        out_grad = output_grads["grad"]
        batch_size = out_grad.shape[0]
        k = self.kernel_size
        group = self.group
        h_out, w_out, num = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, self.n_out_channels
        )
        input_data = self.data

        input_grads = OrderedDict()

        input_grads["grad"] = np.zeros_like(input_data, dtype=dtype)
        self.params["w"].grad = np.zeros_like(
            self.params["w"].value, dtype=dtype)
        self.params["b"].grad = np.zeros_like(
            self.params["b"].value, dtype=dtype)

        for ix in range(batch_size):
            col = self.im2col_conv(input_data[ix], h_in, w_in, c, h_out, w_out)
            col_diff = np.zeros_like(col, dtype=dtype)
            tmp_data_diff = out_grad[ix].reshape((h_out * w_out, num))
            for g in range(group):
                prev_g = g * k * k * c // group
                next_g = (g + 1) * k * k * c // group
                prev_num = g * num // group
                next_num = (g + 1) * num // group
                col_g = col[prev_g: next_g, :]
                weight = self.params["w"].value[:, prev_num: next_num]
                # compute gradients
                self.params["w"].grad[:, prev_num: next_num] += np.dot(
                    col_g, tmp_data_diff[:, prev_num: next_num])
                self.params["b"].grad[:, prev_num: next_num] += \
                    tmp_data_diff[:, prev_num: next_num].sum(0)
                col_diff[prev_g: next_g, :] = np.dot(
                    weight, tmp_data_diff[:, prev_num: next_num].transpose())
            im = self.col2im_conv(
                col_diff.flatten(), h_in, w_in, c, h_out, w_out)
            input_grads["grad"][ix] = im.flatten()
        self.params["w"].grad /= dtype(batch_size)
        self.params["b"].grad /= dtype(batch_size)
        return input_grads
