import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from DataReader import *


b = tf.constant([[2,3],[3,4]])
print(b)
print(b*[3,2])