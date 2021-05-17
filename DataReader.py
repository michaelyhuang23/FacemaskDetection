import numpy as np
import tensorflow as tf
import json
from HelperLib import *

def boxes_to_obj_dataset(all_boxes, all_sizes, validation_split_rate, IoU_threshold):
    total_size = len(all_boxes)

    def evaluate_objectness(r, c, r_size, c_size, box):
        return get_IoU((c*c_size, r*r_size, (c+1)*c_size, (r+1)*r_size), box)+1-IoU_threshold
        # threshold of IoU is 0.3

    def boxes_to_obj(nrow, ncol, boxes):
        frow = calc_preprocessor_output_size(nrow)
        fcol = calc_preprocessor_output_size(ncol)
        box_count = len(boxes)
        calc_func = [calc_2, calc_3, calc_5, calc_8, calc_12]
        maxVal = 0
        fobjectness_list = []
        for i, box in enumerate(boxes):
            maxVal = 0
            objectness_list = []
            for j in range(5):
                rsize = calc_func[j](frow)
                csize = calc_func[j](fcol)
                objectness = np.fromfunction(evaluate_objectness, (rsize, csize),
                                             r_size=nrow/rsize, c_size=ncol/csize, box=box)
                objectness_list.append(objectness)
                maxVal = max(maxVal, np.max(objectness))
            if maxVal < 1:
                objectness_list = [obj*1.001/maxVal for obj in objectness_list]
            # print(max([np.max(obj) for obj in objectness_list]))
            if i == 0:
                fobjectness_list = objectness_list
            else:
                fobjectness_list = [np.maximum(obj1, obj2) for obj1, obj2 in zip(
                    objectness_list, fobjectness_list)]

        fobjectness_list = [np.floor(obj) for obj in fobjectness_list]
        return fobjectness_list

    all_objectnesses = []

    for i in range(total_size):
        ret = boxes_to_obj(*all_sizes[i], all_boxes[i])
        all_objectnesses.append(ret)

    train_size = int(total_size*(1-validation_split_rate))
    val_size = total_size-train_size
    del all_boxes
    del all_sizes

    train_objectnesses = all_objectnesses[:train_size]
    val_objectnesses = all_objectnesses[train_size:]

    train_objectnesses = [[sample[i]
                           for sample in train_objectnesses] for i in range(5)]
    val_objectnesses = [[sample[i]
                         for sample in val_objectnesses] for i in range(5)]
    # flips the dimension so the first dimension is the size and the second the sample

    del all_objectnesses

    train_objectness_dataset0 = tf.data.Dataset.from_generator(
        lambda: train_objectnesses[0], output_types=(tf.float32), output_shapes=(None, None))
    train_objectness_dataset1 = tf.data.Dataset.from_generator(
        lambda: train_objectnesses[1], output_types=(tf.float32), output_shapes=(None, None))
    train_objectness_dataset2 = tf.data.Dataset.from_generator(
        lambda: train_objectnesses[2], output_types=(tf.float32), output_shapes=(None, None))
    train_objectness_dataset3 = tf.data.Dataset.from_generator(
        lambda: train_objectnesses[3], output_types=(tf.float32), output_shapes=(None, None))
    train_objectness_dataset4 = tf.data.Dataset.from_generator(
        lambda: train_objectnesses[4], output_types=(tf.float32), output_shapes=(None, None))

    val_objectness_dataset0 = tf.data.Dataset.from_generator(
        lambda: val_objectnesses[0], output_types=(tf.float32), output_shapes=(None, None))
    val_objectness_dataset1 = tf.data.Dataset.from_generator(
        lambda: val_objectnesses[1], output_types=(tf.float32), output_shapes=(None, None))
    val_objectness_dataset2 = tf.data.Dataset.from_generator(
        lambda: val_objectnesses[2], output_types=(tf.float32), output_shapes=(None, None))
    val_objectness_dataset3 = tf.data.Dataset.from_generator(
        lambda: val_objectnesses[3], output_types=(tf.float32), output_shapes=(None, None))
    val_objectness_dataset4 = tf.data.Dataset.from_generator(
        lambda: val_objectnesses[4], output_types=(tf.float32), output_shapes=(None, None))

    return (train_objectness_dataset0, train_objectness_dataset1, train_objectness_dataset2, train_objectness_dataset3, train_objectness_dataset4), (val_objectness_dataset0, val_objectness_dataset1, val_objectness_dataset2, val_objectness_dataset3, val_objectness_dataset4)


def read_imgs(img_path, validation_split_rate):
    all_imgs_np = np.load(img_path, allow_pickle=True)
    total_size = len(all_imgs_np)
    train_size = int(total_size*(1-validation_split_rate))
    train_imgs_np = all_imgs_np[:train_size]
    val_imgs_np = all_imgs_np[train_size:]
    train_imgs_dataset = tf.data.Dataset.from_generator(
        lambda: train_imgs_np, output_types=tf.float32, output_shapes=(None, None, 3))
    val_imgs_dataset = tf.data.Dataset.from_generator(
        lambda: val_imgs_np, output_types=tf.float32, output_shapes=(None, None, 3))
    return train_imgs_dataset, val_imgs_dataset, train_size, total_size-train_size

def read_boxes(box_path, size_path):
    with open(box_path, 'r') as of:
        all_boxes = json.load(of)
    with open(size_path, 'r') as of:
        all_sizes = json.load(of)
    box_dataset = tf.data.Dataset.from_generator(
        lambda: all_boxes, output_types=(tf.int32), output_shapes=(None,4))
    size_dataset = tf.data.Dataset.from_generator(lambda: all_sizes, output_types=(tf.int32), output_shapes=(2))
    return box_dataset, size_dataset


@tf.function
def random_resize(img,*objectnesses):
    rand_scale = tf.random.uniform(shape=[],minval=0.5,maxval=2)
    target = (int(rand_scale*tf.shape(img)[0]),int(rand_scale*tf.shape(img)[1]))
    new_img = tf.image.resize(img,size=target)
    calc_func = [calc_2, calc_3, calc_5, calc_8, calc_12]
    prep_row = calc_preprocessor_output_size(tf.shape(img)[0])
    prep_col = calc_preprocessor_output_size(tf.shape(img)[1])
    new_objectnesses = [tf.image.resize(obj,size=(calc_func[i](prep_row),calc_func[i](prep_col))) for i,obj in enumerate(objectnesses)]


def read_data(img_path, box_path, size_path, val_split, IoU_threshold):
    print('reading data and preprocessing...')
    train_img_dataset, val_img_dataset, train_size, val_size = read_imgs(img_path, val_split)
    train_box_datasets, val_box_datasets = read_boxes(
        box_path, size_path, val_split, IoU_threshold)

    train_dataset = tf.data.Dataset.zip(
        (train_img_dataset, *train_box_datasets))
    val_dataset = tf.data.Dataset.zip(
        (val_img_dataset, *val_box_datasets))

    train_dataset.map()

    train_dataset = train_dataset.batch(1)
    train_dataset = train_dataset.prefetch(5)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.prefetch(5)
    print('finished')
    return train_dataset, val_dataset, train_size, val_size

