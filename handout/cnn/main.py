from __future__ import division
from collections import OrderedDict
import time
import numpy as np
from data_loader import load_mnist
from tqdm import tqdm
from le_net import LeNet, get_lenet_layers
from optimizer import SGD, SGDMomentum


if __name__ == "__main__":

    layers = get_lenet_layers()

    xtrain, ytrain, xtest, ytest = load_mnist("../Data")

    # Optimization parameters
    opt_params = OrderedDict()
    opt_params["mu"] = 0.9  # momentum
    opt_params["epsilon"] = 0.01  # initial learning rate
    opt_params["gamma"] = 0.0001
    opt_params["power"] = 0.75
    opt_params["weight_decay"] = 0.0005  # weight decay on w

    # display information
    test_interval = 500
    display_interval = 1
    snapshot = 5000
    max_iter = 10000

    # batcher parameters
    batch_size = 64

    lenet = LeNet(layers)
    lenet.save_model("../models/my_model.model")
    optimizer = SGDMomentum(lenet, **opt_params)
    # optimizer = SGD(lenet.parameters(), lr=0.1)
    epochs = 10
    per_epoch = -(-xtrain.shape[0] // batch_size)
    iter_cnt = 0
    for epoch in range(epochs):
        for ix in tqdm(range(per_epoch)):
            optimizer.update_lr(iter_cnt)
            rand_ix = np.random.randint(0, xtrain.shape[0], (batch_size,))
            batch_x = xtrain[rand_ix]
            batch_y = ytrain[rand_ix]
            my_loss = lenet.forward(batch_x.reshape((batch_size, -1)), batch_y)
            lenet.backward(my_loss)
            optimizer.step()
            optimizer.zero_grad()
            iter_cnt += 1
        print("Epoch {0} of {1} Done. Starting Testing".format(
            epoch + 1, epochs))
        start = time.time()
        test_loss = lenet.forward(xtest, ytest)
        print("Testing Done. Took {0:.2f}s. Accuracy: {1:.2f}, Loss: {2:.2f}".format(
            time.time() - start, test_loss["acc"], test_loss["loss"]))
        lenet.save_model("weights.npy")
