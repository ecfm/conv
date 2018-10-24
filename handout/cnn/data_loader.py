from __future__ import division
from scipy.io import loadmat
import numpy as np
import os


def convert_scipy_mat_to_numpy_arrays(filename, output_dir):
    """Converts the matlab mat file to numpy arrays

    Arguments:
        filename (str): The path to the "all_mnist.mat" file
        output_dir (str): The directory to store all the numpy arrays
    Returns:
        None
    """
    data = loadmat(filename)

    xtrain = np.concatenate([data['train{0}'.format(ix)] for ix in range(10)],
                            axis=0)
    ytrain = np.concatenate(
        [ix * np.ones((data['train{0}'.format(ix)].shape[0], 1))
         for ix in range(10)], axis=0
    )
    xtest = np.concatenate([data['test{0}'.format(ix)] for ix in range(10)],
                           axis=0)
    ytest = np.concatenate(
        [ix * np.ones((data['test{0}'.format(ix)].shape[0], 1))
         for ix in range(10)], axis=0
    )
    train_x_filename = os.path.join(output_dir, "train_x.npy")
    train_y_filename = os.path.join(output_dir, "train_y.npy")

    np.save(train_x_filename, xtrain.astype("uint8"))
    np.save(train_y_filename, ytrain.astype("uint8"))

    test_x_filename = os.path.join(output_dir, "test_x.npy")
    test_y_filename = os.path.join(output_dir, "test_y.npy")

    np.save(test_x_filename, xtest.astype("uint8"))
    np.save(test_y_filename, ytest.astype("uint8"))


def load_mnist(data_dir, dtype="float64"):
    """
    Load matrices from data_dir. Return matrices with dtype
    """
    train_x = np.load(os.path.join(data_dir, "train_x.npy")).astype(dtype)
    test_x = np.load(os.path.join(data_dir, "test_x.npy")).astype(dtype)

    train_y = np.load(os.path.join(data_dir, "train_y.npy")).astype(dtype)
    test_y = np.load(os.path.join(data_dir, "test_y.npy")).astype(dtype)

    # Shuffle the data and normalize
    p_ix = np.random.permutation(train_x.shape[0])
    train_x = train_x[p_ix] / 255.
    train_y = train_y[p_ix]

    # Now return
    return train_x, train_y.flatten(), test_x, test_y.flatten()


def process_rotated(data):
    raw_x = data[:, :-1]
    raw_y = data[:, -1]
    p_ix = np.random.permutation(data.shape[0])
    x = raw_x[p_ix]
    y = raw_y[p_ix]
    return x, y


def load_rotated(data_dir, dtype="float32"):
    """
    Loads rotated matrices
    """
    train_x = np.load(os.path.join(data_dir, "rotated_train_x.npy"))
    train_y = np.load(os.path.join(data_dir, "rotated_train_y.npy"))
    test_x = np.load(os.path.join(data_dir, "rotated_test_x.npy"))
    test_y = np.load(os.path.join(data_dir, "rotated_test_y.npy"))
    return train_x, train_y, test_x, test_y


def save_rotated(data_dir, dtype="float32"):
    test_file = "mnist_all_rotation_normalized_float_test.amat"
    train_file = "mnist_all_rotation_normalized_float_train_valid.amat"
    data_train = np.loadtxt(
        os.path.join(data_dir, train_file),
        delimiter=" "
    ).astype(dtype)
    data_test = np.loadtxt(
        os.path.join(data_dir, test_file),
        delimiter=" "
    ).astype(dtype)
    train_x, train_y = process_rotated(data_train)
    test_x, test_y = process_rotated(data_test)
    np.save(os.path.join(data_dir, "rotated_train_x.npy"), train_x)
    np.save(os.path.join(data_dir, "rotated_train_y.npy"), train_y)

    np.save(os.path.join(data_dir, "rotated_test_x.npy"), test_x)
    np.save(os.path.join(data_dir, "rotated_test_y.npy"), test_y)


if __name__ == "__main__":
    base_dir = "../"
    datafile = os.path.join(base_dir, "Data/mnist_all.mat")
    output_dir = "../Data"
    convert_scipy_mat_to_numpy_arrays(datafile, output_dir)
    data_dir = os.path.join(base_dir, "Data")
    xtrain, ytrain, xtest, ytest = load_mnist(data_dir)
