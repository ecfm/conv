from __future__ import division
import numpy as np
from collections import OrderedDict


dtype = np.float32


def softmax(x):
    """Computes the softmax in a numerically stable way
    """
    e_x = np.exp(x - np.max(x, axis=-1).reshape(x.shape[0], 1))
    return e_x / np.sum(e_x, axis=1).reshape(e_x.shape[0], 1)


class LossLayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.params = OrderedDict()

    def forward(self, inputs, return_grad=True, gold_labels=None):
        """Computes the predictions, loss, the accuracy percent, and the gradient.

        Arguments:
            inputs (``OrderedDict``): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array n_in dimensions
            return_grad (bool): Whether to return the gradient wrt the
                softmax loss in the retured dictionary or not
            gold_labels (Union[None, np.ndarray]): The gold labels
                if None is passed, can be used for prediction, otherwise
                can be used to compute the loss and the accuracies.
                Note that the labels array is a (batch,) array, with
                each entry being one of 0, 1, ... 9

       Returns:
            outputs (``OrderedDict``): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                data: the probability predictions (batch, n_classes)
                preds: the predicted class (batch, 1), each element being
                    one of 0, 1, ... 9
                loss (Optional): If the gold labels are specified, then
                    computes the softmax loss
                acc (Optional): If the gold labels are specified, then
                    computes the prediction accuracy
                grad (Optional): If the gold labels are specified, then
                    computes the gradient of the softmax loss wrt the input
                    to this layer

        """
        outputs = OrderedDict()
        for key in inputs:
            if key != "data":
                outputs[key] = inputs[key]
            else:
                # the forward pass
                logits = inputs[key]
                probs = softmax(logits)
                eps = np.finfo(dtype).eps
                outputs["data"] = probs
                preds = np.argmax(probs, 1)
                outputs["preds"] = preds
                if gold_labels is not None:
                    labels = gold_labels.astype(np.int32)
                    categorical_labels = np.zeros(
                        (labels.shape[0], self.n_classes), dtype=dtype)
                    categorical_labels[np.arange(labels.shape[0]), labels] = 1
                    loss = -1 * categorical_labels * np.log(probs + eps)
                    loss = np.sum(loss) / dtype(labels.shape[0])
                    outputs["loss"] = loss
                    if return_grad:
                        outputs["grad"] = probs - categorical_labels
                    accuracy = (labels == preds).sum() / labels.shape[0]
                    outputs["acc"] = accuracy
        return outputs

    def backward(self, **kwargs):
        raise RuntimeError("You should not be calling"
                           "backward on the loss layer")
