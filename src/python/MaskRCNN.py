import tensorflow as tf
tf_config=tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
# import visualize

class Mask:
    """
    """
    def __init__(self, network_name):
        print 'Initializing Mask RCNN network...'
        # Root directory of the project
        ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, network_name + '.h5')

        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        self.classes_to_cut_out = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
        print 'Initialated Mask RCNN network...'

    def GetDynSeg(self,image,image2=None):
        h = image.shape[0]
        w = image.shape[1]
        if len(image.shape) == 2:
            im = np.zeros((h,w,3))
            im[:,:,0]=image
            im[:,:,1]=image
            im[:,:,2]=image
            image = im
        # Run detection
        results = self.model.detect([image], verbose=0)
        # Visualize results
        r = results[0]
        mask = np.zeros((h,w))
        num_masks = len(r['rois'])
        for i in range(num_masks):
            if self.class_names[r['class_ids'][i]] in self.classes_to_cut_out:
                image_m = r['masks'][:,:,i]
                mask[image_m == 1] = 1.
        return mask

