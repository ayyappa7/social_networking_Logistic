import matplotlib.pyplot as plt
import numpy as np


def LogisticCostFunction(expected, output):
    return -1 * (sum(expected * np.log(output + 0.00000001) + (1 - expected) * np.log(1.00000001 - output)) / len(
        expected))


def sigmoid(output):
    return 1 / (1 + np.exp(-1 * output))


def gradientDescent(alpha):
    input = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    params = np.array([[1, 2, 3, 4]])
    expected = np.array([[2], [3]])
    output = np.array([[2], [3]])
    updatedParams = params - alpha * (np.sum(input * (output - expected, 1)))


"""
parameters:
executes the logistic regression for input like
input = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

params = np.array([[1, 2, 3, 4]])
expected = np.array([[2], [3]])
"""
costarr = []


def logistic_regression(input, params, expected, alpha, epochs):
    for i in range(epochs):
        result = np.dot(params, input)
        result = np.transpose(result)
        result = sigmoid(result)
        cost = LogisticCostFunction(expected, result)
        costarr.append(cost)
        params = params - alpha * (np.sum(input * np.transpose((result - expected)), 1))
    return params


# load the training data
file = open("../data/Social_Network_Ads.csv", "r")
train = np.zeros((0, 4))
out = np.zeros((0, 1))
file.readline()
for line in file.readlines():
    words = line.split(",")
    if words[1] == 'Male':
        train = np.append(train, np.array([[1, 1, float(words[2]), float(words[3])]]), axis=0)
    else:
        train = np.append(train, np.array([[1, 0, float(words[2]), float(words[3])]]), axis=0)
    val = float(words[4][0])
    out = np.append(out, np.array([[val]]), axis=0)
train = np.transpose(train)
# scaling
maxOfRows = np.transpose(np.array([np.max(train, axis=1)]))
minOfRows = np.transpose(np.array([np.min(train, axis=1)]))
trainScaled = train - minOfRows
trainScaled[1:] = trainScaled[1:] / (maxOfRows[1:] - minOfRows[1:])

testScaled, trainScaled = np.hsplit(trainScaled, np.array([40]))
out_test = out[:40]
out_train = out[40:]
params = np.array([[0.01, 0.01, 0.01, 0.01]])
alpha = 0.01
params = logistic_regression(trainScaled, params, out_train, alpha, 50)
plt.plot(np.arange(0, len(costarr)), costarr)
plt.show()
testing_out=sigmoid(np.transpose(np.dot(params, testScaled)))
