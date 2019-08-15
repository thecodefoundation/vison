# USAGE: python objectdetected_dictYOLO.py -y yolo-coco -b False

# import the necessary packages
import numpy as np
import time
import cv2
import os
import argparse
import psycopg2
from psycopg2 import Error

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

def establish_connection():
	try:
	    connection = psycopg2.connect(user = "postgres",
	                                  password = "Ayush3186",
	                                  host = "127.0.0.1",
	                                  port = "5432",
	                                  database = "vison_demo_indexdb")
	    cursor = connection.cursor()
	    # Print PostgreSQL Connection properties
	    print (connection.get_dsn_parameters(),"\n")
	    # Print PostgreSQL version
	    cursor.execute("SELECT version();")
	    record = cursor.fetchone()
	    print("You are connected to - ", record,"\n")
	    return connection

	except (Exception, psycopg2.Error) as error :
	    print ("Error while connecting to PostgreSQL", error)

def close_connection(connection):
	if(connection):
		connection.cursor().close()
		connection.close()
		print("PostgreSQL connection is closed")

def insert_into_table(connection, out):

	columns = ', '.join([key.replace(" ", "_") for key in out.keys()])
	values = tuple(out.values())

	try:
		cursor = connection.cursor()
		add_row_query = "INSERT INTO image_index ({0}) VALUES {1}".format(columns, values)
		cursor.execute(add_row_query)
		connection.commit()
  
		print("[INFO] Column added successfully")

	except (Exception, psycopg2.DatabaseError) as error :
	    print ("Error while creating PostgreSQL table", error)

imgPath = 'E:/theCodeFoundation/open_source_images/images_set1/set1/'
images = os.listdir(imgPath)
count = 0
connection = establish_connection()

for image in images:
	if count==20:
		close_connection(connection)
		break

	print('[INFO] Image name: ', image)
	img = cv2.imread(imgPath+image)
	out = detector.runInference(img)
	#next time create table remove id from out. id will be auto increment.
	out.update({'id': count, 'image_id': os.path.splitext(image)[0], 'image_url':'vison{}.com'.format(count)})

	if args["bbox"] is not True:
		print(out)
		insert_into_table(connection, out)
	else:
		cv2.imshow('frame', out)
		cv2.waitKey()

	count+=1
	