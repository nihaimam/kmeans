'''
Niha Imam
CS 484 - 001

implementation of k-means algorithm
'''

import random
import numpy as np
from scipy.spatial import distance

# K-Means Algorithm
def mykmeans(k, data):
    d = len(data[0])  # dimension of data
    max_iter = 100  # no more than 100 to find perfect clusters
    wctr = 0
    # set up clusters; initially set to 1
    curr_cluster = [1] * len(data)  # current designated clusters
    prev_cluster = [-1] * len(data)  # previous designated clusters
    # choose centers for clusters randomly
    centers = []
    for idx in range(0, k):
        centers += [random.choice(data)]
        nogood = False  # in case of bad clusters, recalculate
    # repeat until all clusters have been assigned and all is good
    while (curr_cluster != prev_cluster) or (max_iter > wctr) or (nogood):
        wctr += 1
        print(wctr)
        # since new iteration set current cluster as previous
        prev_cluster = list(curr_cluster)
        nogood = False
        # assign/update each items cluster
        for i in range(0, len(data)):
            minimum = float("inf")  # set the min to arbitrarily small num
            # check if distance from point to center is less
            for j in range(0, len(centers)):
                dist = distance.euclidean(data[i], centers[j])
                # if distance is less, set that as a minimum
                if dist < minimum:
                    minimum = dist
                    curr_cluster[i] = j  # reassign point to new cluster
        # update the cluster center
        for m in range(0, len(centers)):
            temp_center = [0] * d  # create a new center to replace
            points = 0  # number of points in that cluster
            for n in range(0, len(data)):
                # if point belongs to current cluster count number of points
                if curr_cluster[n] == m:
                    for o in range(0, d):
                        temp_center[o] += data[n][o]
                    points += 1
            for p in range(0, d):
                # find the new center by dividing (mean)
                if points != 0:
                    temp_center[p] = temp_center[p] / float(points)
                # if no points in the cluster, select new random center
                else:
                    temp_center = random.choice(data)
                    nogood = True
            # set the center to the temp center / update it
            centers[m] = temp_center

    # compute SSE
    sse = 0.0
    for i in range(0, len(data)):
        dist = distance.euclidean(data[i], centers[curr_cluster[i]])
        sse += dist

    return curr_cluster, sse




random.seed(42)
# empty list for the data
data = []
k = 3  # number of clusters

# read in the data so its a list of n d-dimensional vectors
with open("iris_data.txt", "r") as iris:
    for line in iris:
        p = []
        words = line.split(' ')
        p.append(float(words[0]))
        p.append(float(words[1]))
        p.append(float(words[2]))
        p.append(float(words[3]))
        data.append(p)

# call the method
final_clusters = mykmeans(k, data)

# since clusters have been in 0, 1, 2, add 1
for ctr in range(0, len(final_clusters)):
    final_clusters[0][ctr] = final_clusters[0][ctr] + 1

# save the final cluster to output file
np.savetxt("kmeans.txt", final_clusters[0], fmt="%d")

'''
used the v-score metric to check

from sklearn.metrics.cluster import v_measure_score
corr = []
pred = []

file = open("iris_labels.txt", 'r')
for line in file:
    corr.append(int(line))

file = open("kmeans.txt", 'r')
for line in file:
    pred.append(int(line))

v = v_measure_score(pred, corr)
print(corr)
print(pred)
print(v)
'''

'''
using pseudo code provided by prof sean luke in his lectures
https://cs.gmu.edu/~sean/cs480/uploads/Main/Lecture3.pdf


# K means works through the following iterative process:
# Pick a value for k (the number of clusters to create)
# Initialize k ‘centroids’ (starting points) in your data
# Create your clusters. Assign each point to the nearest centroid.
# Make your clusters better. Move each centroid to the center of its cluster.
# Repeat steps 3–4 until your centroids converge.
'''