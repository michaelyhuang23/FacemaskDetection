import tensorflow as tf
import time

a=tf.random.uniform((1,2,3,4))
b= tf.zeros((1,2,3,4))
metricAcc = tf.keras.metrics.BinaryAccuracy()

print(a)
print(b)

metricAcc.update_state(b, a)

print(metricAcc.result())