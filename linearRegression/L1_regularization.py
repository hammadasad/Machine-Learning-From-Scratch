# Using L1 Regularization aka Lasso Regression to Choose Important Features
# Few Weights that are Non-Zero and many that are 0 (a Sparse Solution)
# J-Lasso = (Sum(n = 1 to number data points) (Y - Yhat)^2) + L1Constant*||weight||
# Taking a derivative of a cost function (dJ / dW) with L1 Norm, you result with constant * sign(w) => Can't solve for w
# Hence we use Gradient Descent

import numpy as np
import matplotlib.pyplot as plt

#Fat matrix
numDataPoints = 50
dimensionality = 50

X = (np.random.random((numDataPoints, dimensionality)) - 0.5 ) * 10

true_w = np.array([1, 0.5, -0.5] + [0]*(dimensionality - 3))

#Add Guassian random noise
Y = X.dot(true_w) + np.random.randn(numDataPoints) * 0.5

costs = []
weight = np.random.randn(dimensionality) / np.sqrt(dimensionality)
learningRate = 0.001
l1_regularizationRate = 10.0

for t in xrange(500):
    Yhat = X.dot(weight)
    delta = Yhat - Y
    weight = weight - learningRate * (X.T.dot(delta) + l1_regularizationRate * np.sign(weight))
    meanSquareError = delta.dot(delta) / numDataPoints
    costs.append(meanSquareError)

plt.plot(costs)
plt.show()

print "final weight is : ", weight

plt.plot(true_w, label = "true Weight")
plt.plot(weight, label = "weight_map")
plt.legend()
plt.show()
