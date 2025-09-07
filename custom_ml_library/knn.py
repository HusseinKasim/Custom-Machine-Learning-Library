import numpy as np
import math

# Test data for development
x = [[3],[4],[6],[8],[9]]
x_two_features = [[1,2],
                  [3,4],
                  [5,6],
                  [7,8],
                  [9,10]]
x_three_features = [[1, 4.55,35], 
                  [1.32, 6.54,44.4], 
                  [5.34, 8.79, 56.7], 
                  [3.56, 10.2, 77.6], 
                  [4,7.8, 67]
                  ]
y = [11.22, 10.65, 10.2, 12, 11]

# Create k-Nearest Neighbors model
def knn(x, y, mode):
    """
    Computes predictions for linear regression using the Normal Equation.
    x: Python list of values of the feature (single or multiple features)
    y: Python list of actual values of target
    mode: 'classification' or 'regression'
    returns NumPy array of predicted values of target
    """
    y_pred = 0 # Temp val

    # Measure distance of input to all training points

    # Pick closest k to it 

    # Regression or Classification based on input
    return y_pred