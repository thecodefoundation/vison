import cv2
import numpy as np


class imHash(object):
    """docstring or imgHash"""
    def __init__(self, hashsize=8):
        self._hashsize = hashsize
        
    def hash_image(self, image):
        # Grayscale image to make hashing invariant to color changes.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize image to make hashing invariant to aspect ration.
        image = cv2.resize(image, (self._hashsize+1, self._hashsize))

        # Compute horizontal gradient based on p[x]>p[x+1], where p is the pixel value and x is the pixel index 
        diff = image[:,:1] > image[:,:-1]
        # Compute hash and return 
        return sum([2**i for (i,gradresult) in enumerate(diff.flatten()) if gradresult])

    def hamming(self, a, b):
    	return bin(int(a) ^ int(b)).count('1')

    def convert_hash(self, hashed):
    	return int(np.array(hashed, dtype='float64'))
