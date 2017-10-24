import argparse
import matplotlib.pyplot as plt;
import numpy as np;
import pymysql as db;

from collections import defaultdict;
from functools import reduce;
from random import sample;

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterate_once", help="Only iterate over the centroids once", action="store_true")
parser.add_argument("-k", "--cluster_num", help="Number of clusters expected", default=3, type=int)
parser.add_argument("-k_def", "--default_centroid", help="Use hard-coded default initial centroids (active only when NOT using -r)", action="store_true")
parser.add_argument("-n", "--random_data_num", help="Number of random data generated", default=10, type=int)
parser.add_argument("-r", "--random_data", help="Use randomly-generated data", action="store_true")
parser.add_argument("-v", "--verbose", help="Print every iteration temporary centroids", action="store_true")

args = parser.parse_args()

def cluster_color(n):
	color_wheel = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'];
	color_pick = 0;
	while color_pick < n:
		yield color_wheel[color_pick % len(color_wheel)];
		color_pick += 1

def print_centroids(centroid_list):
	for cluster_idx, centroid in enumerate(centroid_list):
		print(cluster_idx, ": ", centroid[0], ", ", centroid[1], sep="")

k_value = args.cluster_num;

#Determine the data used from given arguments
dataplot = np.array(list);
if(args.random_data):
	x_axis = np.random.rand(args.random_data_num)
	y_axis = np.random.rand(args.random_data_num)
	dataplot = np.array([[x,y] for x,y in zip(x_axis, y_axis)]);
else:
	dataplot = np.array([[4,2], [12,2], [6,4], [24,3], [5,3],
		[2,5], [8,2], [9,6], [1,3], [17,4]]);


prev_centroids = np.array(list);
cur_centroids = np.array(list);

if(args.default_centroid and not args.random_data):
	cur_centroids = [[17,4],[8,2],[1,3]]
else:
	centroid_candidates = np.unique(dataplot, axis=0);
	init_centroids_pos = sample(range(len(centroid_candidates)), k_value);
	cur_centroids = centroid_candidates[init_centroids_pos,]

print("INITIAL CENTROIDS");
print_centroids(cur_centroids);

iter_counter = 0;

while(not np.array_equal(prev_centroids, cur_centroids)):
	nearest_centroid = np.array([np.argmin([np.linalg.norm(item-centroid) for centroid in cur_centroids]) for item in dataplot]);

	cluster_map = defaultdict(list);
	for cluster, data in zip(nearest_centroid, dataplot):
		cluster_map[cluster].append(data);

	new_centroids = [[key, reduce((lambda x,y: x+y),data)/len(data)] for key, data in cluster_map.items()]
	new_centroids.sort(key=lambda x: x[0]) #x[0]: data cluster

	prev_centroids = cur_centroids;
	cur_centroids = np.array(list(map(lambda cluster_data: cluster_data[1], new_centroids)));

	iter_counter += 1;
	
	if args.verbose:	
		print("Iteration",iter_counter);
		print_centroids(cur_centroids);

	if args.iterate_once:
		break;

print("=======================================");
print("FINAL CENTROIDS");
print_centroids(cur_centroids);
for (centroid_idx, centroid), color in zip(enumerate(cur_centroids), list(cluster_color(len(cur_centroids)))):
	items = cluster_map[centroid_idx];
	plt.scatter([item[0] for item in items],[item[1] for item in items], c=color);
	plt.scatter([centroid[0]],[centroid[1]], c=color, marker="x");

plt.show()