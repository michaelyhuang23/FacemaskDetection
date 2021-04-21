import tensorflow as tf
import time

z = tf.zeros((2,200,300,2,2))
y = tf.RaggedTensor.from_tensor(z)
st = time.time()

for i in range(10000):
    p = tf.RaggedTensor.from_tensor(z)

et1 = time.time()

for i in range(10000):
    p = y.to_tensor()

et2 = time.time()

print(et1-st,et2-et1)