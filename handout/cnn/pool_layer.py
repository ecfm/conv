from __future__ import division
from conv_layer import BaseConvLayer
from collections import OrderedDict
import numpy as np

dtype = np.float32


class PoolLayer(BaseConvLayer):
    def __init__(self, act_type, kernel_size, stride, pad):
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        self.params = OrderedDict()
        super(PoolLayer, self).__init__()

    def init(self, height, width, in_channels):
        """No need to implement this func"""
        pass

    def forward(self, inputs):
        """The forward pass

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
                    Same as the input channels for this layer
                data: a flattened output data array of the form
                    height * width * channel, unrolled in
                    the same way

        You may want to take a look at the im2col_conv and col2im_conv
        functions present in the base class ``BaseConvLayer``

        You may also find it useful to cache the height, width and
        channel of the input image for the backward pass.
        The output heights, widths and channels can be computed
        on the fly using the ``get_output_dim`` function.

        """
        h_in = inputs["height"]
        w_in = inputs["width"]
        c = inputs["channels"]
        data = inputs["data"]
        batch_size = data.shape[0]
        k = self.kernel_size
        h_out, w_out, c = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, c
        )

        # cache for backward pass
        self.h_in, self.w_in, self.c = h_in, w_in, c
        self.data = data
        self.argmax = np.zeros((batch_size, h_out * w_out))

        outputs = OrderedDict()
        outputs["height"] = h_out
        outputs["width"] = w_out
        outputs["channels"] = c
        for ix in range(batch_size):
            # [k * k * c, h_out * w_out]
            col = self.im2col_conv(data[ix], h_in, w_in, c, h_out, w_out)
            # [h_out * w_out, c]
            max_result = np.zeros((h_out * w_out, c), dtype=dtype)
            argmax_result = np.zeros((h_out * w_out, c), dtype=dtype)
            for ic in range(c):
                max_result[:, ic] = col[ic * k * k: (ic + 1) * k * k, :].max(0)
                argmax_result[:, ic] = col[ic * k * k: (ic + 1) * k * k, :].argmax(0)
            outputs["data"][ix] = max_result.flatten()
            self.argmax[ix] = argmax_result.flatten()
        return outputs

    def backward(self, output_grads):
        """The backward pass

        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that we compute the output heights, widths, and
        channels on the fly in the backward pass as well.

        You may want to take a look at the im2col_conv and col2im_conv
        functions present in the base class ``BaseConvLayer``

        """
        input_grads = OrderedDict()
        input_grads["grad"] = np.zeros_like(self.data, dtype=dtype)
        h_in, w_in, c = self.h_in, self.w_in, self.c
        batch_size = self.data.shape[0]
        output_diff = output_grads["grad"]

        k = self.kernel_size

        h_out, w_out, c = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, c
        )
        input_data = self.data

        input_grads = OrderedDict()

        input_grads["grad"] = np.zeros_like(input_data, dtype=dtype)

        for ix in range(batch_size):
            # [h_out * w_out]
            output_diff_col = output_diff[ix]
            # [k * k * c, h_out * w_out]
            input_grad_col = np.zeros((k * k * c, h_out * w_out), dtype=dtype)
            # [h_out * w_out]
            argmax_col = self.argmax[ix]
            # compute gradients
            # [k * k * c, h_out * w_out] x [h_out * w_out, num] = [k * k * c, num]
            for i in range(h_out * w_out):
                input_grad_col[argmax_col[i], i] = output_diff_col[i]
            im = self.col2im_conv(input_grad_col.flatten(), h_in, w_in, c, h_out, w_out)
            input_grads["grad"][ix] = im.flatten()
        return input_grads
