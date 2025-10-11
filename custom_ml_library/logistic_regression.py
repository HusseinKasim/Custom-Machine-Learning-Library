import numpy as np
import math

# Implement (stable) Sigmoid function
def sigmoid(z):
    if z >= 0:   
        return 1/(1+(math.exp(-z)))
    else:
        return (math.exp(z))/(1+math.exp(z))

# Calculate Cross Entropy loss function (Not fully implemented yet)
def cross_entropy_loss(y, y_pred_prob):
    loss = 0
    for index, _ in enumerate(y):
        loss += -((y[index] * math.log(y_pred_prob[index])) + ((1-y[index]) * math.log(1-y_pred_prob[index])))
    return loss/len(y)

# Create a logistic regression model using (batch) gradient descent 
def logistic_reg_gradient(x, y, learning_rate, iterations, m, b):
    """
    Computes predictions for logistic regression using (batch) gradient descent
    x: Python list of lists representing the values of the feature
    y: Python list of actual values of target
    learning rate: the value of the learning rate used for recalculating m and b
    iterations: the number of iterations to train the model for
    m: Python list of slope values
    b: intercept value
    returns NumPy array of predicted class labels and NumPy array of predicted probabilities of class 1
    """

    # Instantiate y_pred probabilities and loss and initialize iteration counter
    y_pred = []
    y_pred_prob = []
    loss = []
    grad_m_list = []
    interation_counter = 0
    threshold = 0.5
    
    while(interation_counter < iterations):
        # Clear predictions and losses
        y_pred_prob.clear()
        loss.clear()
        grad_m_list.clear()
        
        # Initialize db
        db = 0

        # Fill grad_m_list with zeros
        for val in x[0]:
            grad_m_list.append(0)
        
        # Training loop
        for index, _ in enumerate(x):
            # Initialize linear score z and y_pred_prob_temp
            z = 0
            y_pred_prob_temp = 0

            # Calculate z
            for index2, _ in enumerate(x[index]):
                z += m[index2] * x[index][index2] 
            z += b

            # Calculate y_pred
            y_pred_prob_temp = max(1e-15, min(1-1e-15, sigmoid(z)))
            y_pred_prob.append(y_pred_prob_temp)

            # Calculate cross entropy loss
            loss.append(-((y[index] * math.log(y_pred_prob[index])) + ((1-y[index]) * math.log(1-y_pred_prob[index]))))

            # Calculate gradient dm 
            for index2, _ in enumerate(x[index]):
                grad_m_list[index2] += x[index][index2] * (y_pred_prob[index] - y[index])

            # Calculate gradient db
            db += (y_pred_prob[index] - y[index])

        # Update gradient m
        for index, _ in enumerate(m):
            m[index] = m[index] - (learning_rate * (grad_m_list[index]/len(x))) # Uses average gradient

        # Update gradient b
        b = b - (learning_rate * (db/len(x))) # Uses average gradient
        
        # Update iteration counter
        interation_counter+=1

    # Convert predicted probabilities into class labels
    for index, _ in enumerate(y_pred_prob):
        if(y_pred_prob[index] >= threshold):
            y_pred.append(1)
        else:
            y_pred.append(0)

    return np.array(y_pred), np.array(y_pred_prob)
