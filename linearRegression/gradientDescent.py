#Gradient Descent
#Cost to minimize => the square error
# J = (Y - Xw)transpose * (Y-Xw)
# dJ/dW = -2Xtranspose*Y + 2Xtranspose*Xw = 2Xtranspose * (Yhat - Y)
# Instead of setting ^ to 0 and solving for the weight,
# We take small steps in the direction of the gradient until the weight converges to the solution
# 2 is a constant can be dropped since it'll get absorbed into the learning rate
# For the initial weight, it can be drawn from a Guassian centered at 0 with a variance of 1/Dimensionality


import numpy as np
import matplotlib.pyplot as plt

numDataPoints = 10
dimensionality = 3

#X as NxD matrix
X = np.zeros((numDataPoints, dimensionality))
#Set bias terms
X[:, 0] = 1
#First Five elements of the first column to 1's
X[:5, 1] = 1
#Last 5 elements of the 2nd column to 1's
X[5:, 2] = 1

print X

Y = np.array([0] * 5 + [1] * 5)

print Y

#We can't solve this the usual way since Xtranspose * X is a singular matrix.
#We can use Gradient Descent

costs = []

#Ensure variance of 1/dimensionality of weights
weight = np.random.randn(dimensionality) / np.sqrt(dimensionality)
learningRate = 0.001

for t in xrange(1000):
    Yhat = X.dot(weight)
    delta = Yhat - Y
    weight = weight - learningRate * X.T.dot(delta)
    meanSquareError = delta.dot(delta) / numDataPoints
    costs.append(meanSquareError)

plt.plot(costs)
plt.show()

print("The solution is: ", weight)

#Check if prediction is close to the target
plt.plot(Yhat, label = 'prediction')
plt.plot(Y, label = 'target')
plt.legend()
plt.show()
