from ModelCreator import FullProposer
import numpy as np
import tensorflow as tf
import json
import sys
import time

ProposerModel = FullProposer()

print('loading data and preprocessing...')
with open('Data/data_boxes_train.txt', 'r') as of:
    all_boxes = json.load(of)

with open('Data/data_sizes_train.txt', 'r') as of:
    all_sizes = json.load(of)

total_size = len(all_boxes)


def get_area(box):
    return np.maximum(0, box[2] - box[0] + 1) * np.maximum(0, box[3] - box[1] + 1)


def calc_2(input_size):
    return input_size-1


def calc_3(input_size):
    return input_size-2


def calc_5(input_size):
    return (input_size-5)//2+1


def calc_8(input_size):
    return (input_size-8)//3+1


def calc_12(input_size):
    return (input_size-12)//5+1


def calc_preprocessor_output_size(input_size):
    return (((((((((input_size-3)//2+1)-2)-3)//2+1)-2)-3)//2+1)-3)//2+1


def get_IoU(boxes1, boxes2):
    boxInter = (np.maximum(boxes1[0], boxes2[0]), np.maximum(boxes1[1], boxes2[1]),
                np.minimum(boxes1[2], boxes2[2]), np.minimum(boxes1[3], boxes2[3]))
    interArea = get_area(boxInter)
    unionArea = get_area(boxes1)+get_area(boxes2)-get_area(boxInter)
    return interArea/unionArea


def evaluate_objectness(r, c, r_size, c_size, box):
    return get_IoU((c*c_size, r*r_size, (c+1)*c_size, (r+1)*r_size), box)+0.7
    # threshold of IoU is 0.3

def boxes_to_losses(nrow, ncol, boxes):
    frow = calc_preprocessor_output_size(nrow)
    fcol = calc_preprocessor_output_size(ncol)
    box_count = len(boxes)
    calc_func = [calc_2, calc_3, calc_5, calc_8, calc_12]
    maxVal = 0
    fobjectness_list = []
    for i, box in enumerate(boxes):
        maxVal = 0
        objectness_list = []
        for j, size in enumerate([2, 3, 5, 8, 12]):
            rsize = calc_func[j](frow)
            csize = calc_func[j](fcol)
            objectness = np.fromfunction(evaluate_objectness, (rsize, csize),
                                         r_size=size*nrow/rsize, c_size=size*ncol/csize, box=box)
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
    all_objectnesses.append(boxes_to_losses(*all_sizes[i], all_boxes[i]))

validation_split_rate = 0.2
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

all_imgs_np = np.load('Data/imgs_train.npy', allow_pickle=True)
train_imgs_np = all_imgs_np[:train_size]
val_imgs_np = all_imgs_np[train_size:]
train_imgs_dataset = tf.data.Dataset.from_generator(
    lambda: train_imgs_np, output_types=tf.float32, output_shapes=(None, None, 3))
val_imgs_dataset = tf.data.Dataset.from_generator(
    lambda: val_imgs_np, output_types=tf.float32, output_shapes=(None, None, 3))


train_dataset = tf.data.Dataset.zip(
    (train_imgs_dataset, train_objectness_dataset0, train_objectness_dataset1, train_objectness_dataset2, train_objectness_dataset3, train_objectness_dataset4))
train_dataset = train_dataset.batch(1)

val_dataset = tf.data.Dataset.zip(
    (val_imgs_dataset, val_objectness_dataset0, val_objectness_dataset1, val_objectness_dataset2, val_objectness_dataset3, val_objectness_dataset4))
val_dataset = val_dataset.batch(1)

print('finished')

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

