import numpy as np 
import math
import random
import sys

# Test data for development
x_train = [[1,10], [13,11], [15,12], [7,13], [9,10], [4,3], [13,5], [6,5], [8,10], [3,1], [7,12], [10,10], [9,7], [8,8], [10,12], [11, 7]]

# Create a K-means clustering model
def kmeans(x_train, k, iterations):
    """
    Computes predictions for clustering problems using K-means
    x_train: Python list of points of the feature (x,y)
    k: k value
    iterations: Number of iterations used to update centroids
    returns clusters as list(?)
    """

    # Initialize centroids
    initial_centroids = []
    for x in range(k):
        centroid_temp = random.choice(x_train)
        initial_centroids.append((x, centroid_temp))
        x+=1

    current_iteration = 0
    while current_iteration < iterations:

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
                    nearest_centroids.setdefault(str(initial_centroids[index2][0]), []).append(centroids[index][0])

        # Calculate new centroid
        x_vals = []
        new_x = 0
        y_vals = []
        new_y = 0
        final_centroids = []
        for centroid in nearest_centroids.values():
            for index, _ in enumerate(centroid):
                x_val = centroid[index][0]
                x_vals.append(x_val)

                y_val = centroid[index][1]
                y_vals.append(y_val)

            for x in x_vals:
                new_x += x
            new_x = new_x/len(x_vals)

            for y in y_vals:
                new_y += y
            new_y = new_y/len(y_vals)

            x_vals.clear()
            y_vals.clear()

            final_centroids.append([new_x, new_y])

        initial_centroids = []
        for index, centroid in enumerate(final_centroids):
            initial_centroids.append((index, centroid))

        current_iteration+=1

    return initial_centroids


print(kmeans(x_train, 2, 100))
