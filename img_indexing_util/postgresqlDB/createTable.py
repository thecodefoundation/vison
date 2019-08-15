import psycopg2
from psycopg2 import Error

try:
    connection = psycopg2.connect(user = "postgres",
                              password = "Ayush3186",
                              host = "127.0.0.1",
                              port = "5432",
                              database = "vison_demo_indexdb")

    cursor = connection.cursor()
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected to - ", record,"\n")

    create_table_query = '''CREATE TABLE image_index
          (ID SERIAL PRIMARY KEY     NOT NULL,
          image_id           text    NOT NULL,
          image_url         text
          ); '''
    
    cursor.execute(create_table_query)
    connection.commit()

    print("Table created successfully in PostgreSQL ")

except (Exception, psycopg2.DatabaseError) as error :
    print ("Error while creating PostgreSQL table", error)

finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")