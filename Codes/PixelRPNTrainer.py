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

all_objectnesses = [[sample[i] for sample in all_objectnesses] for i in range(5)]
# flips the dimension so the first dimension is the size and the second the sample

validation_split_rate = 0.2
train_size = int(total_size*(1-validation_split_rate))
val_size = total_size-train_size
del all_boxes
del all_sizes

train_objectnesses = all_objectnesses[:][:train_size]
val_objectnesses = all_objectnesses[:][train_size:]

train_objectness_dataset0 = tf.data.Dataset.from_generator(lambda: train_objectnesses[0], output_types=(tf.float32),output_shapes=(None,None))
train_objectness_dataset1 = tf.data.Dataset.from_generator(lambda: train_objectnesses[1], output_types=(tf.float32),output_shapes=(None,None))
train_objectness_dataset2 = tf.data.Dataset.from_generator(lambda: train_objectnesses[2], output_types=(tf.float32),output_shapes=(None,None))
train_objectness_dataset3 = tf.data.Dataset.from_generator(lambda: train_objectnesses[3], output_types=(tf.float32),output_shapes=(None,None))
train_objectness_dataset4 = tf.data.Dataset.from_generator(lambda: train_objectnesses[4], output_types=(tf.float32),output_shapes=(None,None))

val_objectness_dataset0 = tf.data.Dataset.from_generator(lambda: val_objectnesses[0], output_types=(tf.float32),output_shapes=(None,None))
val_objectness_dataset1 = tf.data.Dataset.from_generator(lambda: val_objectnesses[1], output_types=(tf.float32),output_shapes=(None,None))
val_objectness_dataset2 = tf.data.Dataset.from_generator(lambda: val_objectnesses[2], output_types=(tf.float32),output_shapes=(None,None))
val_objectness_dataset3 = tf.data.Dataset.from_generator(lambda: val_objectnesses[3], output_types=(tf.float32),output_shapes=(None,None))
val_objectness_dataset4 = tf.data.Dataset.from_generator(lambda: val_objectnesses[4], output_types=(tf.float32),output_shapes=(None,None))
del all_objectnesses


all_imgs_np = np.load('Data/imgs_train.npy', allow_pickle=True)
train_imgs_np = all_imgs_np[:train_size]
val_imgs_np = all_imgs_np[train_size:]
train_imgs_dataset = tf.data.Dataset.from_generator(
    lambda: train_imgs_np, output_types=tf.float32, output_shapes=(None, None, 3))
val_imgs_dataset = tf.data.Dataset.from_generator(
    lambda: val_imgs_np, output_types=tf.float32, output_shapes=(None, None, 3))


train_dataset = tf.data.Dataset.zip(
    (train_imgs_dataset, train_objectness_dataset0,train_objectness_dataset1,train_objectness_dataset2,train_objectness_dataset3,train_objectness_dataset4))

train_dataset = train_dataset.batch(1)
val_dataset = tf.data.Dataset.zip(
    (val_imgs_dataset, val_objectness_dataset0,val_objectness_dataset1,val_objectness_dataset2,val_objectness_dataset3,val_objectness_dataset4))
val_dataset = val_dataset.batch(1)

print('finished')

@tf.function
def loss_binary_crossentropy(pred0, pred1, pred2, pred3, pred4, label0,label1,label2,label3,label4):
    bce = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM)
    lossSum = bce(pred0[:,:,:,0],label0)
    lossSum += bce(pred1[:,:,:,0],label1)
    lossSum += bce(pred2[:,:,:,0],label2)
    lossSum += bce(pred3[:,:,:,0],label3)
    lossSum += bce(pred4[:,:,:,0],label4)
    return lossSum

@tf.function(input_signature=[tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),])
def train_step(img, label0,label1,label2,label3,label4, optimizer):
    st = time.time()
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        objectnesses = ProposerModel(img, training=True)
        obt = time.time()
        loss = loss_binary_crossentropy(*objectnesses, label0,label1,label2,label3,label4)
        # first_dim of label_objectnesses is batchsize
    el = time.time()
    gradients = tape.gradient(loss, ProposerModel.trainable_variables)
    gett = time.time()
    optimizer.apply_gradients(
        zip(gradients, ProposerModel.trainable_variables))
    aplT = time.time()
    sys.stdout.write("time: %f, %f, %f, %f\r" % (obt-st,el-obt,gett-el,aplT-gett))
    sys.stdout.flush()


adamOptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
Epoch = 1
for epoch in range(Epoch):
    print(f'training epoch {epoch+1}...')
    for i, train_data in enumerate(train_dataset):
        # sys.stdout.write("training: %d/%d\r" % (i,train_size))
        # sys.stdout.flush()
        print(*train_data)
        train_step(*train_data, adamOptimizer)
    lossAvg = 0
    for i,val_data in enumerate(val_dataset):
        img, label_objectnesses = val_data
        objectnesses = ProposerModel(img)
        loss = loss_binary_crossentropy(objectnesses, label_objectnesses[0])
        lossAvg=(lossAvg*i+loss)/(i+1)
    print(f'epoch {epoch+1} is finished')
    print(f'loss value is {lossAvg}')

