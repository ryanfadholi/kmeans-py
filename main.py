import numpy as np 

from simplekmeans import kmeans

k_value = 5
datapoints = np.loadtxt("dataset-uts.txt")
print(f"Type: {type(datapoints)}")
kmeans(k_value, datapoints)