import numpy as np
import tensorflow as tf

def get_area(box):
    return np.maximum(0.0, box[2] - box[0] + 1.0) * np.maximum(0.0, box[3] - box[1] + 1.0)

def calc_2(input_size):
    return input_size-1

def calc_3(input_size):
    return input_size-2

def calc_5(input_size):
    return (((input_size-1)//2+1)-3)//1+1

def calc_8(input_size):
    return (((input_size-1)//2+1)-3)//2+1

def calc_12(input_size):
    return (((input_size-1)//2+1)-6)//3+1

def calc_2_inv(input_size):
    return input_size+1

def calc_3_inv(input_size):
    return input_size+2

def calc_5_inv(input_size):
    return (((input_size-1)*1+3)-1)*2+1

def calc_8_inv(input_size):
    return (((input_size-1)*2+3)-1)*2+1

def calc_12_inv(input_size):
    return (((input_size-1)*3+6)-1)*2+1

calc_func = [calc_2, calc_3, calc_5, calc_8, calc_12]
calc_func_inv = [calc_2_inv, calc_3_inv, calc_5_inv, calc_8_inv, calc_12_inv]

def calc_preprocessor_output_size(input_size):
    return (((((((((input_size-3)//2+1)-2)-3)//2+1)-2)-3)//2+1)-3)//2+1

def get_IoU(boxes1, boxes2):
    boxInter = (np.maximum(boxes1[0], boxes2[0]), np.maximum(boxes1[1], boxes2[1]),
                np.minimum(boxes1[2], boxes2[2]), np.minimum(boxes1[3], boxes2[3]))
    interArea = get_area(boxInter)
    unionArea = get_area(boxes1)+get_area(boxes2)-get_area(boxInter)
    return interArea/unionArea

def count_elements_step(labels, elem_count, positive_count):
    elem_count = [tf.size(labels[i])+elem_count[i] for i in range(5)]
    positive_count = [tf.math.count_nonzero(labels[i])+positive_count[i] for i in range(5)]
    return elem_count, positive_count

def count_elements(labels_data):
    elem_count = [0,0,0,0,0]
    positive_count = [0,0,0,0,0]
    for labels in labels_data:
        elem_count, positive_count = count_elements_step(labels,elem_count,positive_count)
    return positive_count,[tf.cast(elem_count[i],tf.int64)-positive_count[i] for i in range(5)]