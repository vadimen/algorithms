#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:10:03 2020

@author: vadim
"""

import numpy as np
import tensorflow as tf
import pathlib
import cv2
import time

from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

#loader
def load_model(model_name):
	base_url = 'http://download.tensorflow.org/models/object_detection/'
	model_file = model_name + '.tar.gz'
	model_dir = tf.keras.utils.get_file(
			fname=model_name,
			origin=base_url+model_file,
			untar=True)
	
	model_dir = pathlib.Path(model_dir)/"saved_model"
	model = tf.saved_model.load(str(model_dir))
	model = model.signatures['serving_default']
	
	return model

#loading label map
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
		PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

print(detection_model.output_shapes)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict
	
def show_inference(model, image_path):
	image_np = np.array(Image.open(image_path))
	output_dict = run_inference_for_single_image(model, image_np)
	
	vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			category_index,
			instance_masks=output_dict.get('detection_masks_reframed', None),
			use_normalized_coordinates=True,
			line_thickness=8)
	
	#display(Image.fromarray(image_np))

cap = cv2.VideoCapture("video.mp4")

while True:
	ret, image = cap.read()
	
	if ret==True:
		start_time = time.time()
		
		image_np = image
		output_dict = run_inference_for_single_image(detection_model, image_np)
		
		vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
				output_dict['detection_boxes'],
				output_dict['detection_classes'],
				output_dict['detection_scores'],
				category_index,
				instance_masks=output_dict.get('detection_masks_reframed', None),
				use_normalized_coordinates=True,
				line_thickness=8)
		
		print('{:.2f} FPS'.format(1/(time.time()-start_time)))
		#cv2.imshow("image detection", image)
	else: break

	if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cv2.destroyAllWindows()
