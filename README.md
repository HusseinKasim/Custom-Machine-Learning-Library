# Custom Machine Learning Library
---------------------------------
A small Python library made to help me better understand how the classic machine learning algorithms work by implementing them from scratch.

Currently implemented:
1. Linear Regression using Normal Equation (single and multiple features)
2. Linear Regression using Gradient Descent (single and multiple features)
3. Logistic Regression using Gradient Descent (single and multiple features)
4. K-Nearest Neighbors (regression and binary/multiclass classification)
5. K-means

---------------------------------
## Linear Regression
Two types of linear regression were implemented:
  1. Linear Regression using the Normal Equation
  2. Linear Regression using Gradient Descent

### Linear Regression using the Normal Equation
---------------------------------
```
from custom_ml_library import linear_regression

custom_ml_library.linear_regression.linear_reg_normal(x, y)
```
x: Python list of values of the features

y: Python list of actual values of target

returns NumPy array of predicted values of target

---------------------------------
#### Example of implementation using a single feature dataset
```
from custom_ml_library import linear_regression

# Single feature
x = [[3],[4],[6],[8],[9]]
y = [11.22, 10.65, 10.2, 12, 11]

custom_ml_library.linear_regression.linear_reg_normal(x, y)
```

#### Example of implementation using a multiple feature dataset
```
from custom_ml_library import linear_regression

# Multiple features
x_two_features = [[1,2],
                  [3,4],
                  [5,6],
                  [7,8],
                  [9,10]]
y = [11.22, 10.65, 10.2, 12, 11]
custom_ml_library.linear_regression.linear_reg_normal(x_two_features, y)
```

### Linear Regression using Gradient Descent
--------------------------------------------
```
from custom_ml_library import linear_regression

custom_ml_library.linear_regression.linear_reg_gradient(x, y, learning_rate, iterations, m, b)
```
x: Python list of lists representing the values of the feature

y: Python list of actual values of target

learning rate: the value of the learning rate used for recalculating m and b

iterations: the number of iterations to train the model for

m: Python list of initial slope values

b: initial intercept value

returns NumPy array of predicted values of target

---------------------------------

#### Example of implementation using single feature dataset
```
from custom_ml_library import linear_regression

# Single feature
x = [[3],[4],[6],[8],[9]]
y = [11.22, 10.65, 10.2, 12, 11]

custom_ml_library.linear_regression.linear_reg_gradient(x, y, 0.01, 100, [0], 0)
```

#### Example of implementation using multiple feature dataset
```
from custom_ml_library import linear_regression

# Multiple features
x_two_features = [[1,2],
                  [3,4],
                  [5,6],
                  [7,8],
                  [9,10]]
y = [11.22, 10.65, 10.2, 12, 11]

custom_ml_library.linear_regression.linear_reg_gradient(x_two_features, y, 0.01, 100, [0, 0], 0)
```


## Logistic Regression
----------------------
Logistic Regression was implemented using Gradient Descent.

### Logistic Regression using Gradient Descent
----------------------------------------------
```
from custom_ml_library import logistic_regression

custom_ml_library.logistic_regression.logistic_reg_gradient(x, y, learning_rate, iterations, m, b):
```
x: Python list of lists representing the values of the feature

y: Python list of actual values of target

learning rate: the value of the learning rate used for recalculating m and b

iterations: the number of iterations to train the model for

m: Python list of slope values

b: intercept value

returns NumPy array of predicted class labels and NumPy array of predicted probabilities of class 1

------------------------------------------------------

#### Example of implementation using single feature dataset

```
from custom_ml_library import logistic_regression

# Single feature
x = [[3],[4],[6],[8],[9]]
y = [11.22, 10.65, 10.2, 12, 11]

custom_ml_library.logistic_regression.logistic_reg_gradient(x, y, 0.01, 100, [0], 0)
```

#### Example of implementation using multiple feature dataset
```
from custom_ml_library import logistic_regression

# Multiple features
x_two_features = [[1,2],
                  [3,4],
                  [5,6],
                  [7,8],
                  [9,10]]
y = [11.22, 10.65, 10.2, 12, 11]

custom_ml_library.logistic_regression.logistic_reg_gradient(x_two_features, y, 0.01, 100, [0, 0], 0)
```
---------------------------------


