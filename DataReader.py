import numpy as np
import tensorflow as tf
import json
from HelperLib import *

@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.float32)])
def boxes_to_obj(boxes, nrow, ncol, IoU_threshold):
    def evaluate_objectness(r, c, r_size, c_size, box):
        return get_IoU((c*c_size, r*r_size, (c+1)*c_size, (r+1)*r_size), box)+1-IoU_threshold
        # threshold of IoU is 0.3
    frow = calc_preprocessor_output_size(nrow)
    fcol = calc_preprocessor_output_size(ncol)
    maxVal = 0
    fobjectness_list = []
    for box in boxes:
        maxVal = 0
        objectness_list = []
        for j in range(5):
            rsize = calc_func[j](frow)
            csize = calc_func[j](fcol)
            # use tf functions to replace np.fromfunction
            objectness = np.fromfunction(evaluate_objectness, (rsize, csize),
                                            r_size=nrow/rsize, c_size=ncol/csize, box=box)
            objectness_list.append(objectness)
            maxVal = max(maxVal, np.max(objectness))
        if maxVal < 1:
            objectness_list = [obj*1.001/maxVal for obj in objectness_list]
        # print(max([np.max(obj) for obj in objectness_list]))
        if len(fobjectness_list) == 0:
            fobjectness_list = objectness_list
        else:
            fobjectness_list = [np.maximum(obj1, obj2) for obj1, obj2 in zip(
                objectness_list, fobjectness_list)]

    fobjectness_list = [np.floor(obj) for obj in fobjectness_list]
    return fobjectness_list

# def boxes_to_obj_dataset(size, boxes, validation_split_rate, IoU_threshold):
#     def evaluate_objectness(r, c, r_size, c_size, box):
#         return get_IoU((c*c_size, r*r_size, (c+1)*c_size, (r+1)*r_size), box)+1-IoU_threshold
#         # threshold of IoU is 0.3

#     def boxes_to_obj(nrow, ncol, boxes):
#         frow = calc_preprocessor_output_size(nrow)
#         fcol = calc_preprocessor_output_size(ncol)
#         box_count = len(boxes)
#         calc_func = [calc_2, calc_3, calc_5, calc_8, calc_12]
#         maxVal = 0
#         fobjectness_list = []
#         for i, box in enumerate(boxes):
#             maxVal = 0
#             objectness_list = []
#             for j in range(5):
#                 rsize = calc_func[j](frow)
#                 csize = calc_func[j](fcol)
#                 objectness = np.fromfunction(evaluate_objectness, (rsize, csize),
#                                              r_size=nrow/rsize, c_size=ncol/csize, box=box)
#                 objectness_list.append(objectness)
#                 maxVal = max(maxVal, np.max(objectness))
#             if maxVal < 1:
#                 objectness_list = [obj*1.001/maxVal for obj in objectness_list]
#             # print(max([np.max(obj) for obj in objectness_list]))
#             if i == 0:
#                 fobjectness_list = objectness_list
#             else:
#                 fobjectness_list = [np.maximum(obj1, obj2) for obj1, obj2 in zip(
#                     objectness_list, fobjectness_list)]

#         fobjectness_list = [np.floor(obj) for obj in fobjectness_list]
#         return fobjectness_list

#     all_objectnesses = []

#     for i in range(total_size):
#         ret = boxes_to_obj(*all_sizes[i], all_boxes[i])
#         all_objectnesses.append(ret)

#     train_size = int(total_size*(1-validation_split_rate))
#     val_size = total_size-train_size
#     del all_boxes
#     del all_sizes

#     train_objectnesses = all_objectnesses[:train_size]
#     val_objectnesses = all_objectnesses[train_size:]

#     train_objectnesses = [[sample[i]
#                            for sample in train_objectnesses] for i in range(5)]
#     val_objectnesses = [[sample[i]
#                          for sample in val_objectnesses] for i in range(5)]
#     # flips the dimension so the first dimension is the size and the second the sample

#     del all_objectnesses

#     train_objectness_dataset0 = tf.data.Dataset.from_generator(
#         lambda: train_objectnesses[0], output_types=(tf.float32), output_shapes=(None, None))
#     train_objectness_dataset1 = tf.data.Dataset.from_generator(
#         lambda: train_objectnesses[1], output_types=(tf.float32), output_shapes=(None, None))
#     train_objectness_dataset2 = tf.data.Dataset.from_generator(
#         lambda: train_objectnesses[2], output_types=(tf.float32), output_shapes=(None, None))
#     train_objectness_dataset3 = tf.data.Dataset.from_generator(
#         lambda: train_objectnesses[3], output_types=(tf.float32), output_shapes=(None, None))
#     train_objectness_dataset4 = tf.data.Dataset.from_generator(
#         lambda: train_objectnesses[4], output_types=(tf.float32), output_shapes=(None, None))

#     val_objectness_dataset0 = tf.data.Dataset.from_generator(
#         lambda: val_objectnesses[0], output_types=(tf.float32), output_shapes=(None, None))
#     val_objectness_dataset1 = tf.data.Dataset.from_generator(
#         lambda: val_objectnesses[1], output_types=(tf.float32), output_shapes=(None, None))
#     val_objectness_dataset2 = tf.data.Dataset.from_generator(
#         lambda: val_objectnesses[2], output_types=(tf.float32), output_shapes=(None, None))
#     val_objectness_dataset3 = tf.data.Dataset.from_generator(
#         lambda: val_objectnesses[3], output_types=(tf.float32), output_shapes=(None, None))
#     val_objectness_dataset4 = tf.data.Dataset.from_generator(
#         lambda: val_objectnesses[4], output_types=(tf.float32), output_shapes=(None, None))

#     return (train_objectness_dataset0, train_objectness_dataset1, train_objectness_dataset2, train_objectness_dataset3, train_objectness_dataset4), (val_objectness_dataset0, val_objectness_dataset1, val_objectness_dataset2, val_objectness_dataset3, val_objectness_dataset4)


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
    train_img_dataset = read_imgs(img_path)
    train_box_dataset = read_boxes(box_path)
    train_dataset = tf.data.Dataset.zip((train_img_dataset,train_box_dataset))
    train_dataset = train_dataset.repeat(5)
    train_dataset = train_dataset.map(random_resize)
    train_dataset = train_dataset.map(lambda img,boxes : (img,*boxes_to_obj(boxes,tf.shape(img)[0],tf.shape(img)[1],IoU_threshold)))
    print(train_dataset.element_spec)
    # train_dataset = train_dataset.batch(1)
    # train_dataset = train_dataset.prefetch(5)
    # val_dataset = val_dataset.batch(1)
    # val_dataset = val_dataset.prefetch(5)
    print('finished')
    return train_dataset

train_dataset = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt',0.2, 0.3)
for data in train_dataset:
    img, boxes = data
    print(img.shape)
