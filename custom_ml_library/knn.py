import numpy as np
import math

# Test data for development
x_train = [[3, 100],[4, 200],[6, 300],[8, 400],[9, 500]]

y_train = ['b', 'b', 'c', 'c', 'b']

x = [[5, 250]]

# Create k-Nearest Neighbors model
def knn(x_train, y_train, x, k, mode):
    """
    Computes predictions for linear regression using the Normal Equation.
    x: Python list of values of the feature (single or multiple features)
    y: Python list of actual values of target
    mode: 'classification' or 'regression'
    returns NumPy array of predicted values of target
    """
    y_pred = 0 # Temp val
    dist_temp = 0
    dist = {}
    sorted_dist = {}

    # Measure distance of input to all training points
    for index, _ in enumerate(x_train):
        dist_temp = math.sqrt(((x[0][0] - x_train[index][0]) ** 2) + ((x[0][1] - x_train[index][1]) ** 2))
        dist[str(y_train[index])] = dist_temp

    # Sort dict by value
    sorted_dist = dict(sorted(dist.items(), key=lambda y_train:y_train[1]))
    print(sorted_dist)

    # Pick closest k to it


    # Regression or Classification based on input
    if(mode == 'classification'):
      return y_pred
    elif(mode == 'regression'):
      return y_pred
    else:
      ValueError("Mode is incorrect!")

print(knn(x_train, y_train, x, 3, 'classification'))