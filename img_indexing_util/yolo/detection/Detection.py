import numpy as np
import argparse
import imutils
import time
import cv2
import os
import json
import csv
import pandas as pd
from collections import Counter

class yoloDetection:
	# __slots__ = input, path_yolo_files, confidence, threshold, bbox, _LABELS, _COLORS, _net,
	# _ln, _W, _H
	def __init__(self, path_yolo_files, inputPort=0, confidence=0.5, threshold=0.3, bbox=True):
		"""
		inputPort = Input file path or Port number to camera
		path_yolo_files = path to COCO class labels YOLO was trained on. 
							  COCO class labels stored in "coco.names" file.
		confidence = Minimum probability to filter weak detections
		threshold = required for non-maxima supression
		bbox = Enable if you want to see bounding box of detected objects.
		"""
		self.input = inputPort
		self.path_yolo_files = path_yolo_files
		self.confidence = confidence
		self.threshold = threshold
		self.bbox = bbox
		self.label_dict = {}

		self._LABELS = None
		self._COLORS = None
		self._net = None
		self._ln = None
		(self._W, self._H) = (None, None)	

	def getRequiredYOLOfiles(self, *args):
		'''
		Return: '.name', '.weights', '.cfg' files
		
		if *args is 'labelMap' 
		Return: '.csv' file
		'''
		if 'labelMap' in args:
			csv_file = ''
			for file in os.listdir(self.path_yolo_files):
				if file.endswith('.csv'):
					csv_file+=file

			return csv_file

		name_file = ''
		weights_file = ''
		config_file = ''
		for file in os.listdir(self.path_yolo_files):
			if file.endswith(".names"):
				name_file+=file
			if file.endswith(".weights"):
				weights_file+=file
			if file.endswith(".cfg"):
				config_file+=file

		return name_file, weights_file, config_file

	def prepareModel(self, *args):
		"""
		Create model for forward inference.
		return model
		"""
		name_file, weights_file, config_file = self.getRequiredYOLOfiles()

		labelsPath = os.path.sep.join([self.path_yolo_files, name_file])
		self._LABELS = open(labelsPath).read().strip().split("\n")
		np.random.seed(42)
		self._COLORS = np.random.randint(0, 255, size=(len(self._LABELS), 3),
									dtype="uint8")

		### 
		weightsPath = os.path.sep.join([self.path_yolo_files, weights_file])
		configPath = os.path.sep.join([self.path_yolo_files, config_file])

		# load our YOLO object detector trained on COCO dataset (80 classes)
		# and determine only the *output* layer names that we need from YOLO
		print("[INFO] loading YOLO from disk...")
		self._net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
		self._ln = self._net.getLayerNames()
		self._ln = [self._ln[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]
		print("[INFO] loading successful...")

		if args[0] == 'search_detection':
			pass
		else:
			print("[INFO] loading label csv from disk...")
			csv_file = self.getRequiredYOLOfiles('labelMap')
			labelID_label = pd.read_csv(self.path_yolo_files+'/'+csv_file, header=None)
			self.label_dict = labelID_label.T.to_dict('list')
			print("[INFO] loading successful...")

	def runInference(self, frame):
		"""
		Forward pass and object detection
		"""
		if self._W is None or self._H is None:
			(self._H, self._W) = frame.shape[:2]
		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabi lities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		self._net.setInput(blob)

		### Output of the after detection
		layerOutputs = self._net.forward(self._ln)

		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > self.confidence:
					box = detection[0:4] * np.array([self._W, self._H, self._W, self._H])
					(centerX, centerY, width, height) = box.astype("int")

					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
		
		if self.bbox is True:
			frame = self.drawBoundingBox(frame, boxes, confidences, classIDs)
			return frame
		else:
			label_num = dict()
			classID_label = Counter(classIDs)
			for key, value in classID_label.items():
				label_num[self.label_dict[key][0]] = value
			return label_num

	def drawBoundingBox(self, frame, boxes, confidences, classIDs):

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

		if len(idxs) > 0:
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in self._COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(self._LABELS[classIDs[i]],
					confidences[i])
				cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		return frame