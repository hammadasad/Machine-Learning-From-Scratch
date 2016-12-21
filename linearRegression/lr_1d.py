import numpy as np
import matplotlib.pyplot as plt

#load the data
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

#convert X & Y into Numpy Arrays
X = np.array(X)
Y = np.array(Y)

#Plot it
plt.scatter(X, Y)
plt.show()

#Apply equations
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

#Calculate predicted Y
Yhat = a * X + b
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

#calculate R squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print "The r^2 value is: ", r2
