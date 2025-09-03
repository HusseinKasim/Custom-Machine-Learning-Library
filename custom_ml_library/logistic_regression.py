import numpy as np
import math

# Test data for development
x = [1,2,3,4,5]
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
y = [1,0,0,1,0]

# Implement (stable) Sigmoid function
def sigmoid(z):
    if z >= 0:   
        return 1/(1+(math.exp(-z)))
    else:
        return (math.exp(z))/(1+math.exp(z))

# Calculate Cross Entropy loss function (Not implemented yet)
# def cross_entropy_loss(y, y_pred, error)

# Create a logistic regression model using gradient descent 
def logistic_reg_gradient(x, y, learning_rate, iterations, m, b):
    
    # Instantiate y_pred probabilities and error and initialize iteration counter
    y_pred = []
    y_pred_prob = []
    error = []
    dm_list = []
    interation_coutner = 0
    threshold = 0.5
    
    while(interation_coutner < iterations):
        # Clear predictions and errors
        y_pred_prob.clear()
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
            # Initialize linear score z and y_pred_prob_temp
            z = 0
            y_pred_prob_temp = 0

            # Calculate z
            z = m * x[index] + b

            # Calculate y_pred
            y_pred_prob_temp = max(1e-15, min(1-1e-15, sigmoid(z)))
            y_pred_prob.append(y_pred_prob_temp)

            # Calculate error
            error.append(-((y[index] * math.log(y_pred_prob[index])) + ((1-y[index]) * math.log(1-y_pred_prob[index]))))

            # Calculate gradient dm 
            dm += x[index] * (y_pred_prob[index] - y[index])

            # Calculate gradient db
            db += (y_pred_prob[index] - y[index])

        # Update gradient m
        m = m - (learning_rate * dm)

        # Update gradient b
        b = b - (learning_rate * db)
        
        # Update iteration counter
        interation_coutner+=1

    # Print predicted probabilities (ONLY FOR DEVELOPMENT)
    print(y_pred_prob)

    # Convert predicted probabilities into class labels
    for index, _ in enumerate(y_pred_prob):
        if(y_pred_prob[index] >= threshold):
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred


print(logistic_reg_gradient(x, y, 0.01, 100, 0, 0))
