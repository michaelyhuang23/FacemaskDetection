from ModelCreator import FullProposer
from DataReader import read_data
import numpy as np
import tensorflow as tf
import json
import sys
import time

ProposerModel = FullProposer()

train_dataset, val_dataset, train_size, val_size = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt', 'Data/data_sizes_train.txt', 0.2)

adamOptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
bceLoss = tf.keras.losses.BinaryCrossentropy(
    reduction=tf.keras.losses.Reduction.SUM)
metricAcc = tf.keras.metrics.BinaryAccuracy()

@tf.function(input_signature=[tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32)])
def loss_binary_crossentropy(pred0, pred1, pred2, pred3, pred4, label0, label1, label2, label3, label4):
    lossSum = bceLoss(pred0[:, :, :, 0], label0)/tf.cast(tf.size(label0),tf.float32)
    lossSum += bceLoss(pred1[:, :, :, 0], label1)/tf.cast(tf.size(label1),tf.float32)
    lossSum += bceLoss(pred2[:, :, :, 0], label2)/tf.cast(tf.size(label2),tf.float32)
    lossSum += bceLoss(pred3[:, :, :, 0], label3)/tf.cast(tf.size(label3),tf.float32)
    lossSum += bceLoss(pred4[:, :, :, 0], label4)/tf.cast(tf.size(label4),tf.float32)
    return lossSum

@tf.function(input_signature=[tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32)])
def metric_binary_acc(pred0, pred1, pred2, pred3, pred4, label0, label1, label2, label3, label4):
    metricAcc.update_state(label0, pred0[:, :, :, 0])
    metricAcc.update_state(label1, pred1[:, :, :, 0])
    metricAcc.update_state(label2, pred2[:, :, :, 0])
    metricAcc.update_state(label3, pred3[:, :, :, 0])
    metricAcc.update_state(label4, pred4[:, :, :, 0])


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
def train_step(img, label0, label1, label2, label3, label4):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        objectnesses = ProposerModel(img, training=True)
        loss = loss_binary_crossentropy(
            *objectnesses, label0, label1, label2, label3, label4)
        # first_dim of label_objectnesses is batchsize
    gradients = tape.gradient(loss, ProposerModel.trainable_variables)
    adamOptimizer.apply_gradients(
        zip(gradients, ProposerModel.trainable_variables))


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
def eval_step(img, label0, label1, label2, label3, label4):
    objectnesses = ProposerModel(img, training=False)
    loss = loss_binary_crossentropy(*objectnesses, label0, label1, label2, label3, label4)
    metric_binary_acc(*objectnesses, label0, label1, label2, label3, label4)
    return loss


def eval(dataset, size):
    lossAvg = 0
    metricAcc.reset_states()
    for i, data in enumerate(dataset):
        st = time.time()
        loss = eval_step(*data)
        lossAvg = (lossAvg*i+loss)/(i+1)
        sys.stdout.write("evaluating: %d/%d    time per batch: %f \r" %
                            (i, size, time.time()-st))
        sys.stdout.flush()
    return lossAvg, metricAcc.result()
loss, acc = eval(val_dataset,val_size)
print()
print(f'initial loss is {loss}')
print(f'initial acc is {acc}')

Epoch = 5
for epoch in range(Epoch):
    print(f'training epoch {epoch+1}...')
    for i, train_data in enumerate(train_dataset):
        st = time.time()
        train_step(*train_data)
        sys.stdout.write("training: %d/%d    time per batch: %f \r" %
                         (i, train_size, time.time()-st))
        sys.stdout.flush()
    print(f'epoch {epoch+1} is finished')
    loss, acc = eval(val_dataset,val_size)
    print(f'loss is {loss}')
    print(f'acc is {acc}')

