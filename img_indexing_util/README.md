# Instructions to setup

### Description 

1. This will be a simple guide to set things up so that you can contribute to this project. 
2. As of now everything will run locally and the paths are absolute. So you may face some issues but we have tried our best to make this proof of concept modular and easy to build upon.
3. We shall shift to some cloud service soon. (That will further smoothen the process.)

> This README is focused on simple image indexer and some suggestions to improve the current state. 

### Requirements: 

1. We have used `PostgreSQL 11.5.1` which is a powerful, open source object-relational database system. This is currently holding the results created by the indexer. To download PostgreSQL click [here](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads). 

2. We have used `flask` to build python server. `flask_wtf` is used as a flask extension to `WTforms` library. Install the following packages. 

   ```python
   pip install flask
   pip install flask_wtf
   ```

3. `psycopg2` is used as PostgreSQL database adapter for the Python programming language. 

   ```python
   pip install psycopg2
   ```

4. Since the images are indexed we need to detect the objects in that image. For this we built a custom detection module. For this to work without error. Make sure to have the following frameworks/libraries installed. 

   ```
   tensorflow >= 1.13.1
   keras >= 2.2.4
   openCV >= 3.4.1
   imutils >= 0.5.2
   ```

   To install them do the following. (It would be best if you use anaconda. We love anaconda.)

   ```python
   pip install tensorflow
   pip install keras
   pip install opencv-python / conda install opencv
   ```

5. The `YOLO` object detection is using this pretrained weights on 80 classes. Download the following zip file from [here](https://drive.google.com/file/d/1lgA32mpDNcbkPxpE8ISJugAAdStx9JkV/view?usp=sharing).  Once it's downloaded do `extract to this folder`  and copy the folder named `yolo-coco` to `img_indexing_util/yolo`. 

6. The most important thing to an image indexer is image itself. We used Google's open image dataset extended. Download the dataset from [here](https://storage.googleapis.com/openimages/web/extended.html). Download `set 1` along with `image id's` and `image labels`. In the `indexImages.py` script make sure to change the path to images. 

We have tried listing all the requirements. In case something is left out do raise an issue. We will add that to the README. 

### Steps

1. Clone the repo. 
2. Create a conda virtual environment. This will help manage so much going on. 
3. Make sure to cross check from the requirements listed above. 
4. Once everything is ready. Fire your PostgreSQL server. Name the database as `vision_demo_indexdb`. Go to `img_indexing_util/postgresqlDB` and run `testConnection.py` . You will encounter an error while establishing your connection with PostgreSQL. Open the script and change the password in the 4th line of the script. This password is the one which you were asked to set while installing PostgreSQL. Make sure to change the password in all the subsequent scripts. (Sorry for inconvenience. I will create a friendly module to save you from this hassle.)
5. Run `createTable.py` . This will create the table named `image_index`. 
6. Run `addColumns.py`. This will add all the classes that yolo is trained on as the column to `image_index`.
7. Go to `img_indexing_util/yolo` and run `indexImages.py`. To do so use the following command line command `python indexImages.py -y yolo-coco -b False`. Make sure to change the image path to the downloaded dataset. (`imgPath` ) Your database will be populated with data. 
8. Finally run `app.py`. To do so use the following command `python app.py -y yolo-coco -b False`. This will fire your flask server. Go to your browser and type `localhost:5000` on the URL box. You will find a simple form. ![](yolo/images/index.png)

9. Type in something like `person` or `cat` or `toothbrush` and you may get dummy links like. ![](yolo/images/search.png)

### To be done

1. Obviously have better UI. This playground should go with the website. 
2. The entire stuff need to shifted on cloud. 
3. Better database management.
4. Advance indexing algorithm. Advance query algorithm need to be developed or implemented. 
5. This is demo can be used a guide. We should keep building on top of this. Keep implementing and iterating on what's working and what's not. 

Happy Searching :D
