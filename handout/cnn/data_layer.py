from collections import OrderedDict


class DataLayer(object):
    """Records information about the data.

    Note that we pass information around as flat
    layers, unrolled as height * width * channels.
    So this forms the basis of it
    """

    def __init__(self, height, width, channel):
        self.height = height
        self.width = width
        self.channel = channel
        self.params = OrderedDict()

    def forward(self, inputs):
        """The forward pass for the Data Layer
        This layer simply records information, and passes it along

        Arguments:
            inputs (``OrderedDict``): Containing
                information about the data
        """
        outputs = OrderedDict()
        outputs["data"] = inputs["data"]
        outputs["height"] = self.height
        outputs["width"] = self.width
        outputs["channels"] = self.channel
        return outputs

    def backward(self, output_grads):
        raise RuntimeError("You should not call backward for the data layer")
