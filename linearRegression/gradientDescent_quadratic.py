#Gradient Descent
#Moving in the direction of the gradient of a function
#We get closer & closer to the minimum of that function

#Trying out the concept with a quadratic function.
# J = w^2
# dJ/dW = 2w
# Set initial weight to 20, learning rate to 0.1
# For each iteration
# New Weight <- weight - learningRate * [(dJ/dw)(weight)]

import numpy as np

weight = 20
learningRate = 0.1

for i in xrange(100):
    weight = weight - learningRate * 2 * weight
    print("The weight in iteration: ", i, "is : ", weight)
