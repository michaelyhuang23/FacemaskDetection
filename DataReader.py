import numpy as np
import tensorflow as tf
import json
from HelperLib import *

@tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.float32)])
def boxes_to_obj(boxes, nrow, ncol, IoU_threshold):
    def evaluate_objectness(r, c, r_size, c_size, box):
        r_size = tf.cast(r_size,tf.float32)
        c_size = tf.cast(c_size,tf.float32)
        r = tf.cast(r,tf.float32)
        c = tf.cast(c,tf.float32)
        return get_IoU((c*c_size, r*r_size, (c+1)*c_size, (r+1)*r_size), box)+1-IoU_threshold
        # threshold of IoU is 0.3
    frow = calc_preprocessor_output_size(nrow)
    fcol = calc_preprocessor_output_size(ncol)
    maxVal = 0
    fobjectness_list = [-1.0*tf.ones((calc_func[j](frow),calc_func[j](fcol))) for j in range(5)]
    for box in boxes:
        maxVal = 0.0
        objectness_list = [None]*5
        for j in range(5):
            rsize = calc_func[j](frow)
            csize = calc_func[j](fcol)
            objectness = tf.range(rsize*csize,dtype=tf.int32)
            objectness = tf.vectorized_map(lambda ind : evaluate_objectness(ind//csize,ind%csize,r_size=nrow/rsize, c_size=ncol/csize, box=box),objectness)
            objectness = tf.reshape(objectness, (rsize,csize))
            # use tf functions to replace np.fromfunction
            # objectness = np.fromfunction(evaluate_objectness, (rsize, csize),
            #                                 r_size=nrow/rsize, c_size=ncol/csize, box=box)
            objectness_list[j] = objectness
            maxVal = tf.math.maximum(maxVal, tf.reduce_max(objectness))
        print("finish iteration")
        if maxVal < 1:
            objectness_list = [obj*1.001/maxVal for obj in objectness_list]
        # print(max([np.max(obj) for obj in objectness_list]))
        if fobjectness_list[0][0][0] < 0:
            fobjectness_list = objectness_list
        else:
            fobjectness_list = [tf.math.maximum(obj1, obj2) for obj1, obj2 in zip(
                objectness_list, fobjectness_list)]

    fobjectness_list = [tf.floor(obj) for obj in fobjectness_list]
    return fobjectness_list


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


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),tf.TensorSpec(shape=(None,4), dtype=tf.int32)])
def random_resize(img, boxes):
    rand_scale = tf.random.uniform(shape=[],minval=0.5,maxval=2.0)
    target = (tf.floor(rand_scale*tf.cast(tf.shape(img)[0],tf.float32)),tf.floor(rand_scale*tf.cast(tf.shape(img)[1],tf.float32)))
    img = tf.image.resize(img,size=target)
    boxes = tf.floor(rand_scale*tf.cast(boxes,tf.float32))
    return img, boxes

def read_data(img_path, box_path, val_split, IoU_threshold):
    print('reading data and preprocessing...')
    img_dataset = read_imgs(img_path)
    box_dataset = read_boxes(box_path)
    dataset = tf.data.Dataset.zip((img_dataset,box_dataset))
    dataset = dataset.repeat(5)
    dataset = dataset.map(random_resize)
    dataset = dataset.map(lambda img,boxes : (img,*boxes_to_obj(boxes,tf.shape(img)[0],tf.shape(img)[1],IoU_threshold)))
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(5)
    print('finished')
    return dataset

train_dataset = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt',0.2, 0.3)
