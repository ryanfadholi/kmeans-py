import numpy as np 
import matplotlib.pyplot as plt

from collections import defaultdict
from functools import reduce
from random import sample

def cluster_color(n):
	"""Returns a predetermined set of colors for matplotlib scatterplot"""
	color_wheel = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	color_pick = 0
	while color_pick < n:
		yield color_wheel[color_pick % len(color_wheel)]
		color_pick += 1

def print_centroids(centroid_list):
	for cluster_idx, centroid in enumerate(centroid_list):
		print(cluster_idx, ": ", centroid[0], ", ", centroid[1], sep="")

def calculate_average(items):
	"""Calculates average of items list"""
	total = reduce((lambda x,y: x+y), items)
	return total/len(items)

def generate_initial_centroids(k_value, datapoints):
	"""Generates a array of k items as initial centroid value""" 

	#Get all unique values of datapoints 
	#(to avoid multiple centroids with same value)
	centroid_candidates = np.unique(datapoints, axis=0)

	#Generate k random, unique numbers in the range of 0 to unique data,
	#and pick any values indexed by the random numbers from the datapoint 
	#as initial centroids.
	initial_centroid_pos = sample(range(len(centroid_candidates)), k_value)
	return centroid_candidates[initial_centroid_pos]

def show_scatterplot(centroids, cluster_map):
	"""Generates a visualization of clustered data in form of matplotlib scatterplot."""
	cluster_colors = list(cluster_color(len(centroids)))
	for (centroid_idx, centroid), color in zip(enumerate(centroids), cluster_colors):
		items = cluster_map[centroid_idx]
		plt.scatter([item[0] for item in items],[item[1] for item in items], c=color)
		plt.scatter([centroid[0]],[centroid[1]], c=color, marker="x")

	plt.show()

def kmeans(k_value, datapoints):
	"""Calculates k number of clusters from the datapoints given. Datapoints must be a numpy array"""

	k_value = int(k_value)
	
	if type(datapoints) is not np.ndarray:
		raise ValueError("Second parameter (datapoints) must be a Numpy Array")
	data_dimension = len(datapoints[0])

	prev_centroids = np.array(list)
	cur_centroids = generate_initial_centroids(k_value, datapoints) 

	print("INITIAL CENTROIDS")
	print_centroids(cur_centroids)

	iter_counter = 0
	while(not np.array_equal(prev_centroids, cur_centroids)):

		#For every data in the dataset, count its distance to every centroids. 
		dist_to_centroids = []
		for point in datapoints:
			distances = [np.linalg.norm(point-centroid) for centroid in cur_centroids]
			dist_to_centroids.append(np.array(distances))
		dist_to_centroids = np.array(dist_to_centroids)

		# identify the nearest centroid for every data.
		# This will result in a single-dimensional array, where every value 
		# corresponds to the index of nearest centroid from the data.
		nearest_centroid = np.array([np.argmin(distances) for distances in dist_to_centroids])

		# With the index of nearest centroid for every data, 
		# group every data according to its nearest centroid.
		cluster_map = defaultdict(list)
		for cluster, data in zip(nearest_centroid, datapoints):
			cluster_map[cluster].append(data)

		#Calculate the average point of every group, and use it as new centroid values 
		new_centroids = [[key, calculate_average(data)] 
						for key, data in cluster_map.items()]
		new_centroids.sort(key=lambda x: x[0]) #x[0]: data cluster

		#Assign the newly recalculated centroids.
		prev_centroids = cur_centroids
		cur_centroids = np.array(list(data[1] for data in new_centroids))

		iter_counter += 1

	final_centroids = cur_centroids

	print("=======================================")
	print("FINAL CENTROIDS")
	print_centroids(final_centroids)

	#If the data is 2-dimensional, attempt to show a scatterplot of the clustering result.
	if data_dimension == 2:
		show_scatterplot(final_centroids, cluster_map)