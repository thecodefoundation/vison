# USAGE: python search_image.py -i E:/AImodels/101_ObjectCategories/ant/image_0012.jpg -t pickles/tree_0001.pickle -ht pickles/hash_0001.pickle

import cv2
import os
import numpy as np
import vptree
import pickle
import argparse
from imhash.imhash import imHash

imghash = imHash(hashsize=8)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str,
	help="path to query image")
ap.add_argument("-t", "--tree", required=True, type=str,
	help="path to generated VP-Tree")
ap.add_argument("-ht", "--hashtable", required=True, type=str,
	help="path to generated hashtable")
args = vars(ap.parse_args())

print("[INFO] loading VP-Tree and hashes...")
tree = pickle.loads(open(args['tree'], "rb").read())
hashtable = pickle.loads(open(args['hashtable'], "rb").read())

# load the input query image
image = cv2.imread(args['image'])

# compute the hash for the query image, then convert it
queryhash = imghash.hash_image(image)
queryhash = imghash.convert_hash(queryhash)

print(queryhash)

# perform the search
print("[INFO] performing search...")
results = tree.get_all_in_range(queryhash, 10)
results = sorted(results)

print(results)

for (d, h) in results:
	# grab all image paths in our dataset with the same hash
	resultPaths = hashes.get(h, [])
	print("[INFO] {} total image(s) with d: {}, h: {}".format(
		len(resultPaths), d, h))
 
	# loop over the result paths
	for resultPath in resultPaths:
		# load the result image and display it to our screen
		result = cv2.imread(resultPath)
		cv2.imshow("Result", result)
		cv2.waitKey(0)