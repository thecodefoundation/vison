from flask import Flask, render_template, request

import json

import numpy as np
import os
import tensorflow as tf

from collections import defaultdict, Counter
from PIL import Image

from utils import ops
from utils import label_map_util

numClasses = 90  # number of detectable classes

detectionGraph = tf.Graph()
with detectionGraph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(os.path.join('model', 'frozen_inference_graph.pb'), 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map

label_map = label_map_util.load_labelmap(
    os.path.join('data', 'mscoco_label_map.pbtxt'))

categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=numClasses,
                                                            use_display_name=True)

category_index = label_map_util.create_category_index(categories)

def run_inference_onImage(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_scores'] = output_dict['detection_scores'][0]    

        return output_dict

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

import re
import base64
import random
import string

def convertImage(imgData1, filename):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open(filename+'.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))
    

@app.route('/predict/', methods=['POST'])
def predict():

    global category_index, detectionGraph

    filename = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

    print(filename)

    imgData = request.get_data()
    convertImage(imgData, filename)

    image = Image.open(filename+'.png')
    image.load()

    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3]) # 3 is the alpha channel

    background.save(filename+'.jpg', 'JPEG', quality=100)

    image = Image.open(filename+'.jpg')
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis = 0)
    
    output_dict = run_inference_onImage(image_np, detectionGraph)
    
    counter_dict = Counter(output_dict['detection_classes'])
    main_dict = {}
    
    for ID in counter_dict.keys():
        indices = [i for i, x in enumerate(output_dict['detection_classes']) if x == ID]
        for_sum = []
        for index in indices:
            for_sum.append(output_dict['detection_scores'][index])
        prob = sum(for_sum)/len(output_dict['detection_scores'])
        main_dict[list(category_index[ID].values())[1]] = prob*100

    sorted_dict = sorted(main_dict.items(), key=lambda x: x[1], reverse=True)

    r = json.dumps(sorted_dict)

    os.remove(filename+".png")
    os.remove(filename+".jpg")

    return(str(r))
    
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files['file']
    filename = file.filename
    dst = "/".join([target, filename])
    file.save(dst)

    img = Image.open(dst)
    img_np = load_image_into_numpy_array(img)
    # print(img_np.shape)
    img_np_expanded = np.expand_dims(img_np, axis=0)
    # print(img_np_expanded.shape)

    out_dict = run_inference_onImage(img_np, detectionGraph)
    # print(out_dict)
    counter_dict = Counter(out_dict['detection_classes'])
    # print(counter_dict)
    main_dict = dict()

    for ID in counter_dict.keys():
    	indices = [i for i, x in enumerate(out_dict['detection_classes']) if x==ID]
    	for_sum = []
    	for index in indices:
    		for_sum.append(out_dict['detection_scores'][index])
    	prob = sum(for_sum)/len(out_dict['detection_scores'])
    	main_dict[list(category_index[ID].values())[1]] = prob*100

    # print(main_dict)  

    sorted_dict = sorted(main_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dict)

    os.remove(dst)
    r = json.dumps(sorted_dict)
    return(str(r))

if __name__ == "__main__":

    port = int(os.getenv('PORT', 8000))

    app.run(host='0.0.0.0', port=port, debug=True)
    # app.run(debug=True)
