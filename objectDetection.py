## normal imports

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict, Counter
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


## import cv2

## object detection imports

from utils import ops
from utils import label_map_util
#from utils import visualization_utils as vis_util

print('imports done')

## Model

modelName = 'ssd_mobilenet_v1_coco_2017_11_17'
modelFile = modelName + '.tar.gz'
downloadBase = 'http://download.tensorflow.org/models/object_detection/'
path_to_ckpt = modelName + '/frozen_inference_graph.pb'
path_to_labels = os.path.join('data', 'mscoco_label_map.pbtxt')

numClasses = 90

## Download model

opener = urllib.request.URLopener()
opener.retrieve(downloadBase + modelFile, modelFile)
tarFile = tarfile.open(modelFile)

for file in tarFile.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tarFile.extract(file, os.getcwd())

## load frozen tf model into memory
print('model downloaded')

detectionGraph = tf.Graph()
with detectionGraph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

## Loading label map

label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=numClasses,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

##setup ended
print('setup done\n')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

## load data(image)

path_to_images_dir = 'E:/theCodeFoundation/vison/images'
path_to_images = [os.path.join(path_to_images_dir,
                               'image{}.jpg'.format(i)) for i in range(1,
                                        len(os.listdir(path_to_images_dir))+1)]
###### use list dir and append the names with the path name
print('image paths loaded\n')

object_detected_dict = {}

def run_inference_onImage(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            #print(all_tensor_names)
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


img_num = 0

for image_path in path_to_images:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis = 0)
    
    output_dict = run_inference_onImage(image_np, detectionGraph)

##    print('*************************************************************')
##    print(output_dict['detection_scores'])
##    print(output_dict['detection_classes'])
##    print('*************************************************************')
    
    ### use counter get a dict of ids. add values coreesponding
    ### to index of id in scores. mean of values. hurrah!
    
    counter_dict = Counter(output_dict['detection_classes'])
    main_dict = {}
    img_num = img_num + 1
    
    for ID in counter_dict.keys():
        indices = [i for i, x in enumerate(output_dict['detection_classes']) if x == ID]
        for_sum = []
        for index in indices:
            for_sum.append(output_dict['detection_scores'][index])
        prob = sum(for_sum)/len(output_dict['detection_scores'])
        main_dict[list(category_index[ID].values())[1]] = prob*100

    object_detected_dict['image{}'.format(img_num)] = main_dict
    


