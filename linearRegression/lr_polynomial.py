import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_poly.csv'):
    x, y = line.split(',')
    #x is a scalar
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

#Initial plot
#choosing the original value of X, choosing the 2nd column
plt.scatter(X[:, 1], Y)
plt.show()

#Calculate Weights
weight = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, weight)

#Plot the regression
plt.scatter(X[:, 1], Y)

#plt.plot(X[: , 1], Yhat) => the points should be sorted or else the regression line will be hard to read
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

#Testing
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("The value of r^2 is: ", r2)
