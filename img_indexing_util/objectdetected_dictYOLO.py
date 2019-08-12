# USAGE: python objectdetected_dictYOLO.py -y yolo-coco -b False

# import the necessary packages
import numpy as np
import time
import cv2
import os
import argparse

from detection.Detection import yoloDetection

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-i", "--input", required=False,
	help="path to input video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-b", "--bbox", required=False, default=True,
	help="turn on bounding box")
args = vars(ap.parse_args())


detector = yoloDetection(args["yolo"], args["input"], args["confidence"],
	args["threshold"], args["bbox"])

detector.prepareModel()

imgPath = 'images/'
images = os.listdir(imgPath)

# for image in images:
# 	img = cv2.imread(imgPath+image)

# 	out = detector.runInference(img)

# 	if args["bbox"] is True:
# 		cv2.imshow(''.format(image), out)
# 	else:
# 		print(out)

img = cv2.imread(imgPath+images[1])
out = detector.runInference(img)
if args["bbox"] is True:
	cv2.imshow('frame', out)
	cv2.waitKey()
else:
	print(out)
