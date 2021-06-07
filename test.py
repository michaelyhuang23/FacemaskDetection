import tensorflow as tf
import numpy as np
import time

a = tf.constant([3,2,5],dtype=tf.float32)
b = tf.constant([3,4],dtype=tf.float32)
arr = tf.TensorArray(tf.float32,2)
arr.write(0,a)
arr.write(1,b)
print(arr)