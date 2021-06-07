from re import I
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from DataReader import *


train_dataset = read_data('Data/imgs_train.npy', 'Data/data_boxes_train.txt', 0.7, randomize=True)
for img,*obj in train_dataset.take(5):
    plt.figure()
    plt.imshow(img[0])
    plt.figure()
    plt.imshow(obj[0][0],cmap='gray')
    plt.show()
