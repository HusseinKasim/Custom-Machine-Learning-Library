import numpy as np

# Test data for development
x = [3,4,6,8,9]
y = [10,20,30,40,50]

# Create a linear regression model using the Normal Equation (CURRENT IMPLEMENTATION ONLY FOR ONE FEATURE)
def linear_reg_normal_eq(x, y):
    """
    Computes predictions for linear regression using the Normal Equation.
    x: list of values of the feature 
    y: list of actual values of target
    returns NumPy array of predicted values of target
    """

    # Create matrix X of 1s and x values
    X_ones = np.ones(len(x), dtype=int).reshape(-1,1)
    X_x = np.array(x).reshape(-1,1)
    X = np.concatenate((X_ones, X_x), axis=1)

    # Calculate XT . X
    # np.dot(X.T, X)

    # Calculate inverse(XT . X) 
    # np.linalg.inv(np.dot(X.T, X))

    # Calculate XT . y
    y_arr = np.array(y).reshape(-1,1)
    # np.dot(X.T, y_arr)

    # Calculate Theta using Normal Equation inverse(XT . X) . XT . y
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y_arr))

    # Multiply X . Theta to get predicted values
    y_pred = np.dot(X, theta)

    # Return predicted y values as NumPy array
    return y_pred
    
y_pred = linear_reg_normal_eq(x, y)
print(y_pred)