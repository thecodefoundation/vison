import pandas as pd
import psycopg2
from psycopg2 import Error
import os

parent_dir = os.path.dirname(os.getcwd())
labelPath = parent_dir+"/yolo/yolo-coco/labelID_label_mapping.csv"
labels = pd.read_csv(labelPath, header=None)
labels = labels[0].values

try:
    connection = psycopg2.connect(user = "postgres",
                              password = "Ayush3186",
                              host = "127.0.0.1",
                              port = "5432",
                              database = "vison_demo_indexdb")
    cursor = connection.cursor()
    # print ( connection.get_dsn_parameters(),"\n")
    # Print PostgreSQL version
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("[INFO] You are connected to - ", record,"\n")

    for label in labels:
	    add_column = "ALTER TABLE image_index ADD COLUMN {} INT;".format(label.replace(" ", "_"))
	    cursor.execute(add_column)
	    connection.commit()

    print("[INFO] Column added successfully")

except (Exception, psycopg2.DatabaseError) as error :
    print ("Error while creating PostgreSQL table", error)

finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")









