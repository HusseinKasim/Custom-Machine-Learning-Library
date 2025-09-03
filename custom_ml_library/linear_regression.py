import numpy as np

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

# Calculate Mean Squared Error function (Not implemented yet)
# def mean_squared_error(y, y_pred, error)

# Create a linear regression model using (batch) gradient descent 
def linear_reg_gradient(x, y, learning_rate, iterations, m, b):
    """
    Computes predictions for linear regression using (batch) gradient descent
    x: Python list of lists representing the values of the feature
    y: Python list of actual values of target
    learning rate: the value of the learning rate used for recalculating m and b
    iterations: the number of iterations to train the model for
    m: Python list of slope values
    b: intercept value
    returns NumPy array of predicted values of target
    """

    # Instantiate y_pred and error and initialize iteration counter
    y_pred = []
    error = []
    dm_list = []
    interation_counter = 0

    while(interation_counter < iterations):
        # Clear predictions and errors
        y_pred.clear()
        error.clear()
        dm_list.clear()
        
        # Initialize db
        db = 0

        # Fill dm_list with zeros
        for val in x[0]:
                dm_list.append(0)
        
        # Training loop
        for index, _ in enumerate(x):
            # Initialize y_pred_temp
            y_pred_temp = 0

            # Calculate y_pred
            for index2, _ in enumerate(x[index]):
                y_pred_temp += m[index2]*x[index][index2]
            y_pred_temp+=b
            y_pred.append(y_pred_temp)

            # Calculate error
            error.append(y[index] - y_pred[index])

            # Calculate gradient dm
            for index2, _ in enumerate(x[index]):
                dm_list[index2] += 2/len(x) * x[index][index2] * (y_pred[index] - y[index])  
            
            # Calculate gradient db
            db += 2/len(x) * (y_pred[index] - y[index])
        
        # Update m
        for index, _ in enumerate(m):
            m[index] = m[index] - learning_rate * dm_list[index]
        
        # Update b
        b = b - learning_rate * db

        # Update iteration counter
        interation_counter+=1

    return np.array(y_pred)

