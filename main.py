import numpy as np 

from simplekmeans import kmeans

k_value = 5
datapoints = np.loadtxt("example.txt")
kmeans(k_value, datapoints)