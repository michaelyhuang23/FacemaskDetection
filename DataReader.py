import numpy as np
import tensorflow as tf
import json
from HelperLib import *

#@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.float32),tf.TensorSpec(shape=(), dtype=tf.float32),tf.TensorSpec(shape=(4), dtype=tf.float32),tf.TensorSpec(shape=(), dtype=tf.float32)])
# def evaluate_objectness(r, c, r_size, c_size, box, IoU_thres):
#         r_size = tf.cast(r_size,tf.float32)
#         c_size = tf.cast(c_size,tf.float32)
#         r = tf.cast(r,tf.float32)
#         c = tf.cast(c,tf.float32)
#         return get_IoU((c*c_size, r*r_size, (c+1)*c_size, (r+1)*r_size), box)+1-IoU_thres
        # threshold of IoU is 0.3

#@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.float32),tf.TensorSpec(shape=(4), dtype=tf.float32)])
# def get_objectness(frow, fcol, rsize, csize, nrow, ncol, IoU_threshold, box):
#     objectness = tf.range(rsize*csize,dtype=tf.int32)
#     objectness = tf.vectorized_map(lambda ind : evaluate_objectness(r=ind//csize, c=ind%csize, r_size=tf.cast(nrow/rsize,tf.float32), c_size=tf.cast(ncol/csize,tf.float32), box=box, IoU_thres=IoU_threshold),objectness)
#     objectness = tf.reshape(objectness, (rsize,csize))
#     return objectness

def boxes_to_obj(boxes, nrow, ncol, IoU_threshold):
    frow = calc_preprocessor_output_size(nrow)
    fcol = calc_preprocessor_output_size(ncol)
    fobjectness_list = [tf.zeros((calc_func[j](frow),calc_func[j](fcol))) for j in range(5)]
    def evaluate_objectness(r, c, r_size, c_size, box):
        return get_IoU((c*c_size, r*r_size, (c+1.0)*c_size, (r+1.0)*r_size), box)+1.0-IoU_threshold
    for box in boxes:
        maxVal = 0.0
        objectness_list = [None]*5
        for j in range(5):
            rsize = calc_func[j](frow)
            csize = calc_func[j](fcol)
            objectness = tf.range(rsize*csize,dtype=tf.float32)
            objectness = tf.vectorized_map(lambda ind : evaluate_objectness(r=tf.math.floor(ind/csize), c=ind%csize, r_size=nrow/rsize, c_size=ncol/csize, box=box),objectness)
            objectness = tf.reshape(objectness, (rsize,csize))
            objectness_list[j]=objectness
            maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness))
        if maxVal < 1:
            objectness_list = [obj*1.001/maxVal for obj in objectness_list]
        fobjectness_list = [tf.math.maximum(obj1, obj2) for obj1, obj2 in zip(objectness_list, fobjectness_list)]
    fobjectness_list = [tf.math.floor(obj) for obj in fobjectness_list]
    return fobjectness_list

#@tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.float32)])
# def boxes_to_obj(boxes, nrow, ncol, IoU_threshold):
#     frow = calc_preprocessor_output_size(nrow)
#     fcol = calc_preprocessor_output_size(ncol)
#     maxVal = 0
#     fobjectness1 = -1.0*tf.ones((calc_func[0](frow),calc_func[0](fcol)))
#     fobjectness2 = -1.0*tf.ones((calc_func[1](frow),calc_func[1](fcol)))
#     fobjectness3 = -1.0*tf.ones((calc_func[2](frow),calc_func[2](fcol)))
#     fobjectness4 = -1.0*tf.ones((calc_func[3](frow),calc_func[3](fcol)))
#     fobjectness5 = -1.0*tf.ones((calc_func[4](frow),calc_func[4](fcol)))
#     for box in boxes:
#         maxVal = 0.0
#         objectness1 = get_objectness(frow,fcol,calc_func[0](frow),calc_func[0](fcol),nrow,ncol,IoU_threshold,box)
#         objectness2 = get_objectness(frow,fcol,calc_func[1](frow),calc_func[1](fcol),nrow,ncol,IoU_threshold,box)
#         objectness3 = get_objectness(frow,fcol,calc_func[2](frow),calc_func[2](fcol),nrow,ncol,IoU_threshold,box)
#         objectness4 = get_objectness(frow,fcol,calc_func[3](frow),calc_func[3](fcol),nrow,ncol,IoU_threshold,box)
#         objectness5 = get_objectness(frow,fcol,calc_func[4](frow),calc_func[4](fcol),nrow,ncol,IoU_threshold,box)
#         maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness1))
#         maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness2))
#         maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness3))
#         maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness4))
#         maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness5))
#         #for j in range(5):

#             # use tf functions to replace np.fromfunction
#             # objectness = np.fromfunction(evaluate_objectness, (rsize, csize),
#             #                                 r_size=nrow/rsize, c_size=ncol/csize, box=box)
#             # objectness_list[j] = objectness
#             # maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness))
#         if maxVal < 1:
#             objectness1 *= 1.001/maxVal
#             objectness2 *= 1.001/maxVal
#             objectness3 *= 1.001/maxVal
#             objectness4 *= 1.001/maxVal
#             objectness5 *= 1.001/maxVal
#         fobjectness1 = tf.math.maximum(fobjectness1, objectness1)
#         fobjectness2 = tf.math.maximum(fobjectness2, objectness2)
#         fobjectness3 = tf.math.maximum(fobjectness3, objectness3)
#         fobjectness4 = tf.math.maximum(fobjectness4, objectness4)
#         fobjectness5 = tf.math.maximum(fobjectness5, objectness5)
#     fobjectness1 = tf.floor(fobjectness1)
#     fobjectness2 = tf.floor(fobjectness2)
#     fobjectness3 = tf.floor(fobjectness3)
#     fobjectness4 = tf.floor(fobjectness4)
#     fobjectness5 = tf.floor(fobjectness5)
#     return fobjectness1,fobjectness2,fobjectness3,fobjectness4,fobjectness5


def read_imgs(img_path):
    train_imgs_np = np.load(img_path, allow_pickle=True)
    train_imgs_dataset = tf.data.Dataset.from_generator(
        lambda: train_imgs_np, output_types=tf.float32, output_shapes=(None, None, 3))
    return train_imgs_dataset

def read_boxes(box_path):
    with open(box_path, 'r') as of:
        all_boxes = json.load(of)
    box_dataset = tf.data.Dataset.from_generator(
        lambda: all_boxes, output_types=(tf.int32), output_shapes=(None,4))
    return box_dataset


#@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),tf.TensorSpec(shape=(None,4), dtype=tf.int32)])
def random_resize(img, boxes):
    rand_scale = tf.random.uniform(shape=[],minval=0.5,maxval=2.0)
    target = (tf.floor(rand_scale*tf.cast(tf.shape(img)[0],tf.float32)),tf.floor(rand_scale*tf.cast(tf.shape(img)[1],tf.float32)))
    img = tf.image.resize(img,size=target)
    boxes = tf.floor(rand_scale*tf.cast(boxes,tf.float32))
    return img, boxes

def read_data(img_path, box_path, IoU_threshold, randomize=False):
    img_dataset = read_imgs(img_path)
    box_dataset = read_boxes(box_path)
    dataset = tf.data.Dataset.zip((img_dataset,box_dataset))
    if randomize:
        #dataset = dataset.repeat(1)
        dataset = dataset.map(random_resize)
    dataset = dataset.map(lambda img,boxes :(img,*boxes_to_obj(tf.cast(boxes,tf.float32),tf.cast(tf.shape(img)[0],tf.float32),tf.cast(tf.shape(img)[1],tf.float32),IoU_threshold)))
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(5)
    return dataset

train_dataset = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt', 0.3)
