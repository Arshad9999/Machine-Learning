"""A two-layer sigmoid network evaluated on a small matrix example."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(X, W):
    return sigmoid(np.dot(X, W))


def forward_propagation(X, weights1, weights2):
    hidden = forward_pass(X, weights1)
    return forward_pass(hidden, weights2)


if __name__ == "__main__":
    X = np.array([[1, 2, 3]])
    weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    weights2 = np.array([[1, 2, 3]]).T
    print(forward_propagation(X, weights1, weights2))
