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
    
    # Reshape input data based if it is single or multiple features
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

    # Calculate theta using Normal Equation inverse(XT . X) . XT . y (using the Moore-Penrose pseudoinverse function for robustness)
    theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y_arr))

    # Multiply X . theta to get predicted values
    y_pred = np.dot(X, theta)

    # Return predicted y values as NumPy array
    return y_pred


y_pred_single = linear_reg_normal(x, y)
y_pred_three = linear_reg_normal(x_three_features, y)

print(y_pred_single)
print("\n")
print(y_pred_three)