import numpy as np

# Test data for development
x = [3,4,6,8,9]
x_three_features = [[1, 4.55,35], 
                  [1.32, 6.54,44.4], 
                  [5.34, 8.79, 56.7], 
                  [3.56, 10.2, 77.6], 
                  [4,7.8, 67]
                  ]
y = [11.22, 10.65, 10.2, 12, 11]

# Create a linear regression model using the Normal Equation
def linear_reg_normal(x, y):
    """
    Computes predictions for linear regression using the Normal Equation.
    x: Python list of values of the feature (single or multiple features)
    y: Python list of actual values of target
    returns NumPy array of predicted values of target
    """

    # Create matrix of ones for the intercept
    X_ones = np.ones(len(x), dtype=int).reshape(-1,1)
    
    # Reshape input data based on if it is single or multiple features
    if isinstance(x[0], list):
        X_x = np.array(x).reshape(len(x), len(x[0]))
    elif isinstance(x[0], int) or isinstance(x[0], float):
        X_x = np.array(x).reshape(len(x), 1)
    else:
        raise ValueError("Input data type unsupported")

    # Build X matrix
    X = np.concatenate((X_ones, X_x), axis=1)

    # Reshape target values array
    y_arr = np.array(y).reshape(-1,1)

    # Calculate theta using Normal Equation pseudoinverse(XT . X) . XT . y (using the Moore-Penrose pseudoinverse function for robustness)
    theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y_arr))

    # Multiply X . theta to get predicted values
    y_pred = np.dot(X, theta)

    # Return predicted y values as NumPy array
    return y_pred


# Create a linear regression model using gradient descent
def linear_reg_gradient(x, y, learning_rate, iterations=5, m=0, b=0):
    """
    Computes predictions for linear regression using gradient descent
    x: Python list of values of the feature (only supports single features for now)
    y: Python list of actual values of target
    learning rate: the value of the learning rate used for recalculating m and b
    iterations: the number of iterations to train the model for (default value = 5)
    m: slope value (default value = 0)
    b: intercept value (default value = 0)
    returns NumPy array of predicted values of target
    """
     
    # Instantiate y_pred and error and initialize iteration counter
    y_pred = []
    error = []
    interation_coutner = 0

    while(interation_coutner < iterations):
        # Clear predictions and errors
        y_pred.clear()
        error.clear()

        # Initialize dm and db
        dm = 0
        db= 0
        
        for index, val_x in enumerate(x):
            # Calculate y_pred
            y_pred.append(m*x[index]+b)

            # Calculate error
            error.append(y[index] - y_pred[index])

            # Calculate gradients
            dm += 2/len(x) * x[index] * ((m*x[index]+b) - y[index])
            db += 2/len(x) * ((m*x[index]+b) - y[index])

            # For multiple features, the total dm values must be stored in a list since we will have an m value for each feature

        # Update m and b
        m = m - learning_rate * dm
        b = b - learning_rate * db

        # Update iteration counter
        interation_coutner+=1

    return np.array(y_pred)


