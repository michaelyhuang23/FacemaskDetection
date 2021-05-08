from DataReader import read_data
import tensorflow as tf

train_dataset, val_dataset, train_size, val_size = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt', 'Data/data_sizes_train.txt', 0.2, 0.3)

avgRoll = [0,0,0,0,0]
for i,train_data in enumerate(train_dataset):
    img, *labels = train_data
    props = [tf.cast(tf.math.count_nonzero(labels[j]), tf.int32)/tf.size(labels[j]) for j in range(5)]
    avgRoll=[(avgRoll[j]*i+props[j])/(i+1) for j in range(5)]

print([(1/avgRoll[i]).numpy() for i in range(5)])
