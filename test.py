import tensorflow as tf
import time

a = tf.data.Dataset.range(1,4)
b = tf.data.Dataset.range(4,7)
c = tf.data.Dataset.range(7,10)

d = tf.data.Dataset.zip((a,b))

e = tf.data.Dataset.zip((c,*d))

print(list(e.as_numpy_iterator()))