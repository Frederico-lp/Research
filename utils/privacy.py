from sklearn.neighbors import BallTree
from math import dist

import numpy as np

def get_nearest(src_point, candidates):
    min_dist = 1000
    min_index = 0
    for i in range(len(candidates)):
        distance = dist(src_point, candidates[i])
        if distance < min_dist:
            min_dist = dist
            min_index = i

    return min_index, min_dist

def avg_dist(src_points, candidates):
    sum_dist = []
    for i in range(len(src_points)):
        index, dist = get_nearest(src_points[i], candidates)
        sum_dist += dist

    return sum_dist/i 

    

# def get_nearest(src_points, candidates, k_neighbors=1):
#     '''
#     Find nearest neighbors for all source points from a set of candidate points
#     '''
#     # Create tree from the candidate points
#     knn = NearestNeighbors(n_neighbors=3)
#     #tree = BallTree(candidates, leaf_size=15)
#     # Find closest points and distances
#     distances, indices = tree.query(src_points, k=k_neighbors)
#     # Transpose to get distances and indices into arrays
#     distances = distances.transpose()
#     indices = indices.transpose()
#     # Get closest indices and distances (i.e. array at index 0)
#     # note: for the second closest points, you would take index 1, etc.
#     closest = indices[0]
#     closest_dist = distances[0]
#     # Return indices and distances
#     return (closest, closest_dist)