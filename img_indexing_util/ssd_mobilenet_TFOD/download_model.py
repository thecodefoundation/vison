## normal imports

import os
import shutil
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

print('IMPORTS DONE')

if os.path.isdir('model'):
    shutil.rmtree('model')

## Model

model_loc = 'model'
modelName = 'ssd_mobilenet_v1_coco_2017_11_17'
modelFile = modelName + '.tar.gz'
downloadBase = 'http://download.tensorflow.org/models/object_detection/'
path_to_ckpt = model_loc + '/frozen_inference_graph.pb'
path_to_labels = os.path.join('data', 'label_map.pbtxt')

## Download model

opener = urllib.request.URLopener()
opener.retrieve(downloadBase + modelFile, modelFile)
tarFile = tarfile.open(modelFile)

print('MODEL DOWNLOADED')

for file in tarFile.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tarFile.extract(file, os.getcwd())
    if os.path.isdir('ssd_mobilenet_v1_coco_2017_11_17'):
        os.rename('ssd_mobilenet_v1_coco_2017_11_17', model_loc)
        shutil.rmtree('ssd_mobilenet_v1_coco_2017_11_17')
        
print('FROZEN GRAPH EXTRACTED')
