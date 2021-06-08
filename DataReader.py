import numpy as np
import tensorflow as tf
import cv2
import json

from tensorflow.python.framework.tensor_spec import TensorSpec
from HelperLib import *


def boxes_to_obj(boxes, nrow, ncol, IoU_threshold):
    frow = calc_preprocessor_output_size(nrow)
    fcol = calc_preprocessor_output_size(ncol)
    fobjectness_list = [0]*5
    def evaluate_objectness(r, c, r_size, c_size, box):
        return get_IoU((c*c_size, r*r_size, (c+1)*c_size, (r+1)*r_size), box)+1-IoU_threshold
    for box in boxes:
        maxVal = 0.0
        objectness_list = [None]*5
        for j in range(5):
            rsize = calc_func[j](frow)
            csize = calc_func[j](fcol)
            objectness = np.fromfunction(evaluate_objectness,(rsize, csize),r_size=nrow/rsize, c_size=ncol/csize, box=box)
            objectness_list[j]=objectness
            maxVal = np.maximum(maxVal, np.max(objectness))
        if maxVal < 1:
            objectness_list = [obj*1.001/maxVal for obj in objectness_list]
        fobjectness_list = [np.maximum(obj1, obj2) for obj1, obj2 in zip(objectness_list, fobjectness_list)]
    fobjectness_list = [np.floor(obj) for obj in fobjectness_list]
    return fobjectness_list


def read_imgs(img_path):
    train_imgs_np = np.load(img_path, allow_pickle=True)
    return train_imgs_np

def read_boxes(box_path):
    with open(box_path, 'r') as of:
        all_boxes = json.load(of)
    all_boxes_np = [np.array(boxes) for boxes in all_boxes]
    return all_boxes_np

def random_resize(img, boxes):
    # there's no point going belong 0.7 since there the largest box covers the entire screen
    rand_scale = np.random.uniform(0.7,2.0)
    target = (int(rand_scale*img.shape[1]),int(rand_scale*img.shape[0]))
    img = cv2.resize(img,dsize=target)
    boxes = np.floor(rand_scale*boxes.astype(np.float32))
    return img, boxes

def read_data(img_path, box_path):
    img_data = read_imgs(img_path)
    box_data = read_boxes(box_path)
    return list(zip(img_data,box_data))

# sizes = [(img.shape[0],img.shape[1]) for img in img_data]
#     objs = [[],[],[],[],[]]
#     for size,boxes in zip(sizes,box_data):
#         obj = boxes_to_obj(boxes,*size,IoU_threshold)
#         for i in range(5):
#             objs[i].append(obj[i])
