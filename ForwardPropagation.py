import numpy as np
def sigmoid(x):
  return 1 / (1 + np.exp(-x))]

def forward_pass(X, W):
  output = sigmoid(np.dot(X, W))
  return output

# Input data
X = np.array([[1, 2, 3]])

# Weights for the first layer
weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Weights for the output layer
weights2 = (np.array([[1, 2, 3]])).T

result = forward_propagation(X, weights1, weights2)
print(result)
