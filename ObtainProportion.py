from DataReader import *
import tensorflow as tf
import time
IoU_threshold = 0.3
train_dataset = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt')

avgRoll = [0,0,0,0,0]
repetition = 5
st = time.time()
for k in range(repetition):
    for i,train_data in enumerate(train_dataset):
        img, boxes = train_data
        img, boxes = random_resize(img, boxes)
        labels = boxes_to_obj(boxes, img.shape[0],img.shape[1],IoU_threshold)
        props = [tf.cast(tf.math.count_nonzero(labels[j]), tf.int32)/tf.size(labels[j]) for j in range(5)]
        sz = i+k*len(train_dataset)
        avgRoll=[(avgRoll[j]*sz+props[j])/(sz+1) for j in range(5)]

print([(1/avgRoll[i]).numpy() for i in range(5)])
print(time.time()-st)