'''
Niha Imam
CS 484 - 001

use implementation of k-means algorithm to cluster the MNIST data
'''

import pandas as pd
import numpy as np
from kmeans import mykmeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

# start time to see how long program takes
start_time = time.time()

# read in the data as a csv file
data = pd.read_csv("test_data.txt", header = None)

# checking to see the dimensionality of the data
pca = PCA(n_components=784).fit(data)

# code to plot out the variance
#plt.figure(figsize=(10,6))
#plt.scatter(list(range(784)), pca.explained_variance_ratio_)
#plt.title("Exlained Variance")
#plt.show()

# use pca for dimensionality reduction to 40 dimensions
pca = PCA(n_components=40)
# fit and transform the data
pcaft = pca.fit(data).transform(data)

# used the pca fit data to get the labels and the computed sse
pred = mykmeans(10,pcaft)

# since the k means labels the data from 0-9 we add 1 to every labesl
labels = pred[0]
out = []
for i in range(0, len(labels)):
    out.append(labels[i] + 1)

# write the predicted labels to the txt file
np.savetxt("image.txt", out, fmt="%d")

# find and plot the sse scores for k = 2,4,6,8,10,12,14,16,18,20
# below code plots the sse of the given k values
#x = [2,4,6,8,10,12,14,16,18,20]
#y = []
#for i in range(2, 21, 2):
#    tmp = mykmeans(i,pcaft)
#    y.append(tmp[1])

#plt.plot(x, y)
#plt.xlabel('k')
#plt.ylabel('sse score')
#plt.title('k vs sse score')
#plt.show()

# end time
print("--- %s seconds ---" % (time.time() - start_time))