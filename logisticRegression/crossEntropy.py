import numpy as np
import matplotlib.pyplot as plt

numSamples = 100
numFeatures = 2

X = np.random.randn(numSamples, numFeatures)

X[:50, :] = X[:50, :] - 2 * np.ones((50, numFeatures))
X[50:, :] = X[50:, :] + 2 * np.ones((50, numFeatures))

target = np.array([0] * 50 + [1] * 50)

ones = np.array([[1] * numSamples]).T
Xb = np.concatenate((ones, X), axis = 1)

#initialize weights
weights = np.random.randn(numFeatures + 1)

#calcuate output of the model
output = Xb.dot(weights)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(output)

def calc_cross_entropy(target, Y):
    entropy = 0
    for i in xrange(numSamples):
        if target[i] == 1:
            entropy -= np.log(Y[i])
        else:
            entropy -= np.log(1 - Y[i])
    return entropy

print calc_cross_entropy(target, Y)

# Use the closed form solution for logistic regression => Equal variances of both classes (numpy default for rand)
# Weights only depend on the mean

# weights = np.array([0, 4, 4]) #bias is 0, both weights are 4
weights = np.random.randn(numFeatures + 1)

output = Xb.dot(weights)
Y = sigmoid(output)

# print calc_cross_entropy(target, Y)

# Use gradient descent to optimize weights
learningRate = 0.1
for i in xrange(100):
    if (i % 10 == 0):
        print calc_cross_entropy(target, Y)
    targetDifference = target - Y
    weights += learningRate * targetDifference.T.dot(Xb)
    Y = sigmoid(Xb.dot(weights))

print("Final weight: ", weights)

#Visualizing the bayes classifier

plt.scatter(X[:, 0], X[:, 1], c = target, s = 100, alpha = 0.5)

x_axis = np.linspace(-6 , 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()

# Cross Entropy should be much lower when using the closed form solution
