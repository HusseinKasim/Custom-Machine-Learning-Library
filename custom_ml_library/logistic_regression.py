import numpy as np
import math

# Test data for development
x = [3]
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
y = [0]

# Implement Sigmoid function
def sigmoid(z):
    return 1/(1+(math.e ** -z))

# Calculate Cross Entropy loss function (Not implemented yet)
# def cross_entropy_loss(y, y_pred, error)

# Create a logistic regression model using gradient descent 
def logistic_reg_gradient(x, y, learning_rate, iterations, m, b):
    
    # Instantiate y_pred and error and initialize iteration counter
    y_pred = []
    error = []
    dm_list = []
    interation_coutner = 0
    
    while(interation_coutner < iterations):
        # Clear predictions and errors
        y_pred.clear()
        error.clear()
        dm_list.clear()
        
        # Initialize dm and db
        dm = 0
        db = 0

        # Fill dm_list with zeros
        #for val in x[0]:
        #        dm_list.append(0)
        
        # Training loop
        for index, _ in enumerate(x):
            # Initialize linear score z and y_pred_temp
            z = 0
            y_pred_temp = 0

            # Calculate z
            z = m * x[index] + b

            # Calculate y_pred
            y_pred_temp = sigmoid(z)
            y_pred.append(y_pred_temp)

            # Calculate error
            error.append(-((y[index] * math.log(y_pred[index])) + ((1-y[index]) * math.log(1-y_pred[index]))))

            # Calculate gradients dm and db

        # Update gradients m and b

        # Update iteration counter
        interation_coutner+=1

    return y_pred_temp


print(logistic_reg_gradient(x, y, 0.01, 100, 0, 0))
