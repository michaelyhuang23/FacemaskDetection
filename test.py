import tensorflow as tf
import numpy as np
import time

a = tf.constant([[3,4],[1,2],[3,5]],dtype=tf.float32)
print(tf.reduce_max(a))