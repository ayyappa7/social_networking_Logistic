import numpy as np


def LogisticCostFunction(expected, output):
    return -(sum(expected * np.log(output) + (1 - expected) * np.log(1 - output)) / len(expected))


def sigmoid(output):
    return 1 / (1 + np.exp(-output))


"""
parameters:
executes the logistic regression for input like
input = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

params = np.array([[1, 2, 3, 4]])
expected = np.array([[2], [3]])
"""


def logistic_regression(input, params, expected, costFunction):
    output = np.dot(params, input)
    output = np.transpose(output)
    output = sigmoid(output)
    cost = costFunction(expected, output)
    print(cost)


input = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
params = np.array([[1, 2, 3, 4]])
expected = np.array([[2], [3]])
logistic_regression(input, params, expected, MSE)
