from DataReader import read_data
import tensorflow as tf

train_dataset, val_dataset, train_size, val_size = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt', 'Data/data_sizes_train.txt', 0.2, 0.27)

avgRoll = 0
for i,train_data in enumerate(train_dataset):
    img, label0, label1, label2, label3, label4 = train_data
    prop0 = tf.cast(tf.math.count_nonzero(label0), tf.int32)/tf.size(label0)
    prop1 = tf.cast(tf.math.count_nonzero(label1), tf.int32)/tf.size(label1)
    prop2 = tf.cast(tf.math.count_nonzero(label2), tf.int32)/tf.size(label2)
    prop3 = tf.cast(tf.math.count_nonzero(label3), tf.int32)/tf.size(label3)
    prop4 = tf.cast(tf.math.count_nonzero(label4), tf.int32)/tf.size(label4)
    avgRoll=(avgRoll*i+(prop0+prop1+prop2+prop3+prop4)/5)/(i+1)

print(1/avgRoll)