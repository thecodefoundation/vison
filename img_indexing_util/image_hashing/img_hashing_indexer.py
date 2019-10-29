# USAGE: python img_hashing_indexer.py -i E:/AImodels/101_ObjectCategories -t pickles/tree_0001.pickle -ht pickles/hash_0001.pickle

import cv2
import os
import numpy as np
import vptree
import pickle
import argparse
from imhash.imhash import imHash


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, type=str,
	help="root path to image dataset")
ap.add_argument("-t", "--tree", required=True, type=str,
	help="path to store VP-Tree")
ap.add_argument("-ht", "--hashtable", required=True, type=str,
	help="path to store hashtable")
args = vars(ap.parse_args())

# Absolute path now
datasetPath = args['dataset']

# Dictionary of hashes
hashtable = {}

# Call image hashing utilities
imghash = imHash(hashsize=8)

print('[INFO] Hashing and indexing images...')

for (dirpath, dirnames, filenames) in os.walk(datasetPath):

	if len(filenames)!=0:
		files = [dirpath+'/{}'.format(file) for file in filenames] # add check for image files

		for file in files:
			print('[INFO] Hashing file: {}'.format(file))
			# do hashing and indexing here.
			img = cv2.imread(file)

			if img is None:
				continue

			hashed = imghash.hash_image(img)
			hashed = imghash.convert_hash(hashed)

			l = hashtable.get(hashed, [])
			l.append(file)
			hashtable[hashed] = l

print('[INFO] Generating VP-Tree...')
points = list(hashtable.keys())
tree = vptree.VPTree(points, imghash.hamming)
	
# serialize the VP-Tree to disk
print("[INFO] serializing VP-Tree...")
f = open(args['tree'], "wb")
f.write(pickle.dumps(tree))
f.close()
 
# serialize the hashes to dictionary
print("[INFO] serializing hashes...")
f = open(args['hashtable'], "wb")
f.write(pickle.dumps(hashtable))
f.close()
