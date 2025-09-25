import numpy as np
import math

# Example data
'''
x_train = [[3, 100],[4, 200],[6, 300],[8, 400],[9, 500]]

y_train_regression = [10,20,30,40,50]

y_train_classification = ['b','a','c','b','d']

x = [[5, 250]]
'''

# Create a k-Nearest Neighbors (KNN) model
def knn(x_train, y_train, x, k, mode):
    """
    Computes predictions for linear / BINARY classification problems using KNN
    x_train: Python list of points of the feature (x,y)
    y_train: Python list of actual values (regression)/classes (classification) of target
    x: Python list of new point to predict
    k: k value
    mode: 'classification' or 'regression'
    returns value (regression) / class label (classification) of y_pred for the new point 
    """
    single_dist = 0
    dist_temp = ()
    dist = []
    sorted_dist = []
    closest_k = []

    # Measure distance of input to all training points
    for index, _ in enumerate(x_train):
        single_dist = math.sqrt(((x[0][0] - x_train[index][0]) ** 2) + ((x[0][1] - x_train[index][1]) ** 2))
        dist_temp = (single_dist, y_train[index])
        dist.append(dist_temp)

    # Sort list of tuples by value
    sorted_dist = sorted(dist)

    # Pick closest k to it
    for index, val in enumerate(iter(sorted_dist)):
      if index < k:
        closest_k.append(val)
    
    # Regression or Classification based on input
    if(mode == 'classification'):
      return knn_classification_helper(y_train, closest_k)
    elif(mode == 'regression'):
     return knn_regression_helper(closest_k)
    else:
      ValueError("Mode is incorrect!")

# Helper function for KNN classification
def knn_classification_helper(y_train, closest_k):
  """
  Predicts the class label for the new point
  y_train: Target values of training data
  closest_k: List of tuples of the closest k neighbors
  returns the predicted class label for the new point
  """
  class_label_counters = {}

  # Initialize class labels dict
  for label in y_train:
    if label in class_label_counters.keys():
      continue
    class_label_counters[label] = 0
  
  # Fill class labels dict
  for val in closest_k:
    if val[1] in class_label_counters.keys():
      class_label_counters[val[1]]+=1
  
  return max(class_label_counters, key=class_label_counters.get)

# Helper function for KNN regression
def knn_regression_helper(closest_k):
  """
  Predicts the target value for the new point
  closest_k: List of tuples of the closest k neighbors
  returns the predicted target value for the new point
  """
  total = 0
  for val in closest_k:
    total += val[1]
  y_pred = total/len(closest_k)
  return y_pred

print(knn(x_train, y_train_regression, x, 3, 'regression'))

print(knn(x_train, y_train_classification, x, 3, 'classification'))
