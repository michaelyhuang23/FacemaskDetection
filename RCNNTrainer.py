from ModelCreator import FullRCNN, FullRCNNModel
from DataReader import *
import numpy as np
import tensorflow as tf
import json
import sys
import time

train_size = 650
Epoch = 30
IoU_threshold = 0.3

print('reading data and preprocessing...')
train_data = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt', 'Data/data_types_train.txt')
train_size = len(train_data)
val_data = read_data('Data/imgs_val.npy','Data/data_boxes_val.txt', 'Data/data_types_val.txt')
val_size = len(val_data)

val_data = [(img,*boxes_to_obj(boxes,img.shape[0],img.shape[1],IoU_threshold), boxes, types) for img, boxes, types in val_data]
# shape: img, objs, boxes, types

train_data = [(img,*boxes_to_obj(boxes,img.shape[0],img.shape[1],IoU_threshold), boxes, types) for img, boxes, types in train_data]

sparse_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()
adamOptimizer = tf.keras.optimizers.Adam(learning_rate=3*1e-4)
classification_coeff = 1
regressor_coeff = 1

print("finish load")

def loss_func(r_size, c_size, types, regresses, filters, objboxes ,rtypes, rboxes):
    if tf.shape(types)[0]==0:
        return 0
    filters = filters[:,1:]
    print(objboxes.shape)
    print(tf.shape(filters))
    print(tf.shape(types))
    print(tf.shape(regresses))
    ids = tf.gather_nd(objboxes, filters)
    coboxes = tf.cast(tf.gather_nd(rboxes,ids),tf.float32)
    cotypes = tf.gather_nd(rtypes,ids)
    filters = tf.cast(tf.roll(filters,1,axis=1),tf.float32)
    posLow = filters*tf.constant([c_size,r_size])
    posHigh = (filters+1)*tf.constant([c_size,r_size])
    pos = tf.concat([posLow,posHigh],axis=1)
    regresses*=[1,1,-1,-1]*regressor_coeff
    pos+=regresses
    abs_diff = (pos-coboxes)
    losses = tf.where(abs_diff<1,(abs_diff**2)/2,abs_diff-0.5)
    loss = tf.math.reduce_sum(losses)
    loss += classification_coeff*sparse_crossentropy(cotypes,types)
    return loss


def train_step(img, objs, objboxes, boxes, types):
    nrow = img.shape[0]
    ncol = img.shape[1]
    frow = calc_preprocessor_output_size(nrow)
    fcol = calc_preprocessor_output_size(ncol)
    rsize = [0,0,0,0,0]
    csize = [0,0,0,0,0]
    for i in range(5):
        rsize[i] = calc_func[i](frow)
        csize[i] = calc_func[i](fcol)
    objs = [obj[np.newaxis,...] for obj in objs]
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        ret = FullRCNNModel(img[np.newaxis,...], objs, training=True)
        # shape of ret[0]: class, regress, filter
        loss = 0
        for i,elem in enumerate(ret):
            loss += loss_func(nrow/rsize[i],ncol/csize[i],*elem,objboxes[i],types,boxes)
    gradients = tape.gradient(loss, FullRCNNModel.trainable_variables)
    adamOptimizer.apply_gradients(
        zip(gradients, FullRCNNModel.trainable_variables))
    return loss


def eval_step(img, objs, objboxes, boxes, types):
    objs = [obj[np.newaxis,...] for obj in objs]
    ret = FullRCNNModel(img[np.newaxis,...], objs, training=False)
    nrow = img.shape[0]
    ncol = img.shape[1]
    frow = calc_preprocessor_output_size(nrow)
    fcol = calc_preprocessor_output_size(ncol)
    rsize = [0,0,0,0,0]
    csize = [0,0,0,0,0]
    for i in range(5):
        rsize[i] = calc_func[i](frow)
        csize[i] = calc_func[i](fcol)
    loss = 0
    for i,elem in enumerate(ret):
        print(*elem)
        loss += loss_func(nrow/rsize[i],ncol/csize[i],*elem,objboxes[i],types,boxes)
    return loss


def eval(dataset, size):
    lossAvg = 0
    for i, data in enumerate(dataset):
        st = time.time()
        img, objs, objboxes, boxes, types = data
        loss = eval_step(img,objs,objboxes,boxes,types)
        lossAvg = (lossAvg*i+loss)/(i+1)
        sys.stdout.write("evaluating: %d/%d;  eval loss is: %f;  time per batch: %f \r" %
                         (i, size, lossAvg, time.time()-st))
        sys.stdout.flush()
    return lossAvg


min_loss = eval(val_data, val_size)
print()
print(f'initial loss is {min_loss}')

train_losses = []
train_val_losses = []
for epoch in range(Epoch):
    print(f'training epoch {epoch+1}...')
    lossAvg = 0
    all_labels = []
    for i, data in enumerate(train_data):
        st = time.time()
        img, objs, objboxes, boxes, types = data
        loss = train_step(img,objs,objboxes,boxes,types)
        lossAvg = (lossAvg*i + loss)/(i+1)
        sys.stdout.write("training: %d/%d; train loss is: %f; time per batch: %f \r" %
                        (i, train_size, lossAvg, time.time()-st))
        sys.stdout.flush()
    train_losses.append(lossAvg)
    print(f'epoch {epoch+1} is finished, computing acc...')
    loss = eval(val_data, val_size)
    train_val_losses.append(loss)
    print(f'val loss is {loss}')
    if loss<min_loss:
        FullRCNNModel.save_weights("models/RCNN_train_weights/")
