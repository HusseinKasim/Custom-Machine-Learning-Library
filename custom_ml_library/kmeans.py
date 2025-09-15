import numpy as np 
import math
import random

# Test data for development
x_train = [[1,10], [13,11], [15,12], [7,13], [9,10], [4,3], [13,5], [6,5], [8,10], [3,1], [7,12], [10,10], [9,7], [8,8], [10,12], [11, 7]]

# Create a K-means clustering model
def kmeans(x_train, k):
    """
    Computes predictions for clustering problems using K-means
    x_train: Python list of points of the feature (x,y)
    k: k value
    returns clusters as list(?)
    """
    # Initialize centroids
    initial_centroids = []
    for x in range(k):
        centroid_temp = random.choice(x_train)
        initial_centroids.append(('centroid_' + str(x), centroid_temp))
        x+=1

    # Calculate distance to centroids
    centroids = []
    for index, _ in enumerate(x_train):
        dist = []
        for c_index, _ in enumerate(initial_centroids):
            dist.append(math.sqrt(((initial_centroids[c_index][1][0] - x_train[index][0]) ** 2) + ((initial_centroids[c_index][1][1] - x_train[index][1]) ** 2)))
        centroids.append((x_train[index], dist)) 

    # Select nearest centroids
    nearest_centroids = {}
    for index, val in enumerate(centroids):
        for index2 in range(k):
            if min(val[1]) == val[1][index2]:
                nearest_centroids.setdefault(str(initial_centroids[index2][0]), []).append(str(centroids[index][0]))
    
    print(nearest_centroids)
        
    y_pred = 0 # Temp y_pred
    return y_pred

kmeans(x_train, 3)