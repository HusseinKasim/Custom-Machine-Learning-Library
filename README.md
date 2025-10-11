# Custom Machine Learning Library
---------------------------------
A small Python library made to help me better understand how the classic machine learning algorithms work by implementing them from scratch.

Currently implemented:
1. Linear Regression using Normal Equation (single and multiple features)
2. Linear Regression using Gradient Descent (single and multiple features)
3. Logistic Regression using Gradient Descent (single and multiple features)
4. K-Nearest Neighbors (regression and binary/multiclass classification)
5. K-Means

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


## K-Nearest Neighbors (KNN)
----------------------
KNN implementation 

```
from custom_ml_library import knn

custom_ml_library.knn.knn(x_train, y_train, x, k, mode):
```
x_train: Python list of points of the feature (x,y)

y_train: Python list of actual values (regression)/classes (classification) of target

x: Python list of new point to predict

k: k value

mode: 'classification' or 'regression'

returns value (regression) / class label (classification) of y_pred for the new point 

------------------------------------------------------

#### Example of implementation for regression

```
from custom_ml_library import knn

x_train = [[3, 100],[4, 200],[6, 300],[8, 400],[9, 500]]
y_train_regression = [10,20,30,40,50]
x = [[5, 250]]

custom_ml_library.knn.knn(x_train, y_train_regression, x, 3, 'regression')
```

#### Example of implementation for classification
```
from custom_ml_library import knn

x_train = [[3, 100],[4, 200],[6, 300],[8, 400],[9, 500]]
y_train_classification = ['b','a','c','b','d']
x = [[5, 250]]

custom_ml_library.knn.knn(x_train, y_train_classification, x, 3, 'classification')
```


## K-Means
----------------------
K-Means implementation

```
from custom_ml_library import kmeanms

custom_ml_library.kmeanms.kmeanms(x_train, k, iterations)
```
x_train: Python list of points of the feature (x,y)

k: k value

iterations: Number of iterations used to update centroids

returns centroids as list

------------------------------------------------------

#### Example of implementation

```
from custom_ml_library import kmeans

x_train = [[1,10], [13,11], [15,12], [7,13], [9,10], [4,3], [13,5], [6,5], [8,10], [3,1], [7,12], [10,10], [9,7], [8,8], [10,12], [11, 7]]

custom_ml_library.kmeans.kmeans(x_train, 2, 100)
```
---------------------------------
