from flask import Flask, render_template
from flask import redirect
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
import numpy as np
import time
import cv2
import os
import argparse
import psycopg2
from psycopg2 import Error

from detection.Detection import yoloDetection

app = Flask(__name__)

app.config['SECRET_KEY'] ='AYUSH'

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


def search_table(connection, search_query):
	print("May search")
	try:
		cursor = connection.cursor()
		select_query = "SELECT image_url from image_index where {0} is not null".format(search_query)
		cursor.execute(select_query)
		results = cursor.fetchall()
  
		print("[INFO] Column added successfully")
		return results

	except (Exception, psycopg2.DatabaseError) as error :
	    print ("Error while creating PostgreSQL table", error)

class MyForm(FlaskForm):
    name = StringField('Search: ', validators=[DataRequired()])

@app.route('/', methods=('GET', 'POST'))
def index():
	form = MyForm()
	if form.validate_on_submit():
		connection = establish_connection()
		search_query = form.name.data
		print(search_query)
		results = search_table(connection, search_query)
		print(results)
		close_connection(connection)
	return render_template('home.html', form=form)

if __name__ == '__main__':

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

	detector.prepareModel('search_detection')

	app.run(debug=True)