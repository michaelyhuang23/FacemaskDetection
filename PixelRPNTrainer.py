from ModelCreator import FullProposer
from DataReader import *
from HelperLib import count_elements
import numpy as np
import tensorflow as tf
import json
import sys
import time

ProposerModel = FullProposer()
train_size = 650
Epoch = 30
IoU_threshold = 0.3

print('reading data and preprocessing...')
train_data = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt')
train_size = len(train_data)
val_data = read_data('Data/imgs_val.npy','Data/data_boxes_val.txt')
val_size = len(val_data)

adamOptimizer = tf.keras.optimizers.Adam(learning_rate=3*1e-4)
bceLoss = tf.nn.weighted_cross_entropy_with_logits
metricFalsePos = [tf.keras.metrics.FalsePositives() for i in range(5)]
metricFalseNeg = [tf.keras.metrics.FalseNegatives() for i in range(5)]
loss_positive_scale = 1.5
case_proportions = [142.13843280068323, 127.41335978629806, 52.51596434586317, 35.644690504416815, 30.839078674934328]
# this loss_positive_scale controls the trade off between positive acc and negative acc
val_data = [(img,*boxes_to_obj(boxes,img.shape[0],img.shape[1],IoU_threshold)) for img, boxes in val_data]
# convert to obj data

print("finish load")
val_img, *val_labels = val_data
del val_img
val_positive_count, val_negative_count = count_elements(val_labels)
val_positive_count = [tf.cast(val_positive_count[i], tf.float32) for i in range(5)]
val_negative_count = [tf.cast(val_negative_count[i], tf.float32) for i in range(5)]


@tf.function(input_signature=[tf.TensorSpec(shape=(1,None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32)])
def loss_binary_crossentropy(pred0, pred1, pred2, pred3, pred4, label0, label1, label2, label3, label4):
    lossSum = tf.reduce_sum(bceLoss(label0,
                                    pred0[:, :, :, 0], loss_positive_scale*case_proportions[0]))/tf.cast(tf.size(label0), tf.float32)
    lossSum += tf.reduce_sum(bceLoss(label1, pred1[:, :, :, 0],
                                     loss_positive_scale*case_proportions[1]))/tf.cast(tf.size(label1), tf.float32)
    lossSum += tf.reduce_sum(bceLoss(label2, pred2[:, :, :, 0],
                                     loss_positive_scale*case_proportions[2]))/tf.cast(tf.size(label2), tf.float32)
    lossSum += tf.reduce_sum(bceLoss(label3, pred3[:, :, :, 0],
                                     loss_positive_scale*case_proportions[3]))/tf.cast(tf.size(label3), tf.float32)
    lossSum += tf.reduce_sum(bceLoss(label4, pred4[:, :, :, 0],
                                     loss_positive_scale*case_proportions[4]))/tf.cast(tf.size(label4), tf.float32)
    return lossSum


@tf.function(input_signature=[tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32)])
def metric_binary_acc(pred0, pred1, pred2, pred3, pred4, label0, label1, label2, label3, label4):
    metricFalsePos[0].update_state(label0, tf.math.sigmoid(pred0[:, :, :, 0]))
    metricFalsePos[1].update_state(label1, tf.math.sigmoid(pred1[:, :, :, 0]))
    metricFalsePos[2].update_state(label2, tf.math.sigmoid(pred2[:, :, :, 0]))
    metricFalsePos[3].update_state(label3, tf.math.sigmoid(pred3[:, :, :, 0]))
    metricFalsePos[4].update_state(label4, tf.math.sigmoid(pred4[:, :, :, 0]))

    metricFalseNeg[0].update_state(label0, tf.math.sigmoid(pred0[:, :, :, 0]))
    metricFalseNeg[1].update_state(label1, tf.math.sigmoid(pred1[:, :, :, 0]))
    metricFalseNeg[2].update_state(label2, tf.math.sigmoid(pred2[:, :, :, 0]))
    metricFalseNeg[3].update_state(label3, tf.math.sigmoid(pred3[:, :, :, 0]))
    metricFalseNeg[4].update_state(label4, tf.math.sigmoid(pred4[:, :, :, 0]))


@tf.function(input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None), dtype=tf.float32)])
def train_step(img, label0, label1, label2, label3, label4):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        objectnesses = ProposerModel(img, training=True)
        loss = loss_binary_crossentropy(
            *objectnesses, label0, label1, label2, label3, label4)
        # first_dim of label_objectnesses is batchsize
    gradients = tape.gradient(loss, ProposerModel.trainable_variables)
    metric_binary_acc(*objectnesses, label0, label1, label2, label3, label4)
    adamOptimizer.apply_gradients(
        zip(gradients, ProposerModel.trainable_variables))
    return loss


@tf.function(input_signature=[tf.TensorSpec(shape=(1,None, None, 3), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32), tf.TensorSpec(shape=(1,None, None), dtype=tf.float32)])
def eval_step(img, label0, label1, label2, label3, label4):
    objectnesses = ProposerModel(img, training=False)
    loss = loss_binary_crossentropy(
        *objectnesses, label0, label1, label2, label3, label4)
    metric_binary_acc(*objectnesses, label0, label1, label2, label3, label4)
    return loss


def eval(dataset, size):
    lossAvg = 0
    for i in range(5):
        metricFalsePos[i].reset_states()
        metricFalseNeg[i].reset_states()
    for i, data in enumerate(dataset):
        st = time.time()
        img, label0, label1, label2, label3, label4 = data
        loss = eval_step(img[np.newaxis,...], label0[np.newaxis,...], label1[np.newaxis,...], label2[np.newaxis,...], label3[np.newaxis,...], label4[np.newaxis,...])
        lossAvg = (lossAvg*i+loss)/(i+1)
        sys.stdout.write("evaluating: %d/%d;  eval loss is: %f;  time per batch: %f \r" %
                         (i, size, lossAvg, time.time()-st))
        sys.stdout.flush()
    neg_acc = [(1-metricFalsePos[i].result()/val_negative_count[i]).numpy() for i in range(5)]
    pos_acc = [(1-metricFalseNeg[i].result()/val_positive_count[i]).numpy() for i in range(5)]
    return lossAvg, neg_acc,pos_acc

print('positive acc indicates the percent of actual positives it correctly predicted')
print('negative acc indicates the percent of actual negatives it correctly predicted')

min_loss, negAcc, posAcc = eval(val_data, val_size)
print()
print(f'initial loss is {min_loss}')
print('initial positive acc is',*posAcc)
print('initial negative acc is',*negAcc)
train_losses = []
train_val_losses = []
for epoch in range(Epoch):
    print(f'training epoch {epoch+1}...')
    lossAvg = 0
    for i in range(5):
        metricFalsePos[i].reset_states()
        metricFalseNeg[i].reset_states()
    all_labels = []
    for i, data in enumerate(train_data):
        st = time.time()
        img, boxes = data
        img, boxes = random_resize(img, boxes)
        labels = boxes_to_obj(boxes, img.shape[0],img.shape[1],IoU_threshold)
        labels = [label[np.newaxis,...] for label in labels]
        loss = train_step(img[np.newaxis,...],*labels)
        lossAvg = (lossAvg*i + loss)/(i+1)
        sys.stdout.write("training: %d/%d; train loss is: %f; time per batch: %f \r" %
                        (i, train_size, lossAvg, time.time()-st))
        sys.stdout.flush()
        all_labels.append(labels)
    train_losses.append(lossAvg)
    print(f'epoch {epoch+1} is finished, computing acc...')
    train_positive_count, train_negative_count = count_elements(all_labels)
    train_positive_count = [tf.cast(train_positive_count[i], tf.float32) for i in range(5)]
    train_negative_count = [tf.cast(train_negative_count[i], tf.float32) for i in range(5)]
    neg_acc = [(1-metricFalsePos[i].result()/train_negative_count[i]).numpy() for i in range(5)]
    pos_acc = [(1-metricFalseNeg[i].result()/train_positive_count[i]).numpy() for i in range(5)]
    print('train positive acc is',*pos_acc)
    print('train negative acc is',*neg_acc)
    loss, neg_acc, pos_acc = eval(val_data, val_size)
    train_val_losses.append(loss)
    print(f'val loss is {loss}')
    print('val positive acc is',*pos_acc)
    print('val negative acc is',*neg_acc)
    if loss<min_loss:
        ProposerModel.save_weights("models/PixelRPN_train_weights/")



# Epoch = 15
# ProposerModel.set_layers_trainability(True, 6,ProposerModel.block6_layer_counts)
# adamOptimizer.learning_rate = 1e-4
# finetune_losses = []
# finetune_val_losses = []

# for epoch in range(Epoch):
#     print(f'finetuning epoch {epoch+1}...')
#     lossAvg = 0
#     for i in range(5):
#         metricFalsePos[i].reset_states()
#         metricFalseNeg[i].reset_states()
#     for i, train_data in enumerate(val_dataset):
#         st = time.time()
#         loss = train_step(*train_data)
#         lossAvg = (lossAvg*i + loss)/(i+1)
#         sys.stdout.write("finetuning: %d/%d; train loss is: %f; time per batch: %f \r" %
#                          (i, val_size, lossAvg, time.time()-st))
#         sys.stdout.flush()
#     finetune_losses.append(lossAvg)
#     print(f'epoch {epoch+1} is finished')
#     neg_acc = [(1-metricFalsePos[i].result()/train_negative_count[i]).numpy() for i in range(5)]
#     pos_acc = [(1-metricFalseNeg[i].result()/train_positive_count[i]).numpy() for i in range(5)]
#     print('finetune positive acc is',*pos_acc)
#     print('finetune negative acc is',*neg_acc)
#     loss, neg_acc, pos_acc = eval(val_dataset, val_size)
#     finetune_val_losses.append(loss)
#     print(f'val loss is {loss}')
#     print('val positive acc is',*pos_acc)
#     print('val negative acc is',*neg_acc)
#     if loss<min_loss:
#         ProposerModel.save_weights("models/PixelRPN_train_weights/")

