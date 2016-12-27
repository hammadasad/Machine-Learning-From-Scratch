# Using a sigmoid (activation function) to calculate the output of a neuron

import numpy as np

#Create dummy data

numberDataPoints = 100;
dimensionality = 2;

#normally distributed
X = np.random.randn(numberDataPoints, dimensionality)

#Create & Add Bias
ones = np.array([[1] * numberDataPoints]).T

Xb = np.concatenate((ones, X), axis = 1)

weight = np.random.randn(dimensionality + 1)
z = Xb.dot(weight)

# Soft Step Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print sigmoid(z)
