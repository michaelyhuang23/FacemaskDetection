from os import name
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Input, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from HelperLib import *

full_preprocessor = InceptionV3(
    include_top=False, weights='imagenet')
preprocessor = tf.keras.Model(
    full_preprocessor.input, full_preprocessor.get_layer('mixed7').output)

class PixelRPN(Model):
    def __init__(self):
        super(PixelRPN, self).__init__()
        # input size is about 17*17*768 but larger
        self.dropout_condense1 = Dropout(0.5)
        self.conv2d_condense1 = Conv2D(
            192, (1, 1), padding='same', activation='relu')
        self.batch_norm1 = BatchNormalization()
        # if we set this number too small, it creates information bottleneck
        self.dropout_condense2 = Dropout(0.5)
        self.conv2d_condense2 = Conv2D(
            96, (1, 1), padding='same', activation='relu')
        self.batch_norm2 = BatchNormalization()
        # note that the effective receptive field most likely cover entire image here
        # so every single feature position has the potential to produce bounding box anywhere in the image

        self.conv2d_extract2a = Conv2D(
            96, (2, 2), padding='same', activation='relu')
        self.batch_norm_extract2a = BatchNormalization()
        self.conv2d_extract2b = Conv2D(96, (2,2), padding='valid', activation='relu')
        self.batch_norm_extract2b = BatchNormalization()
        self.dropout_extract2 = Dropout(0.35)
        self.conv2d_extract2c = Conv2D(96, (1, 1), activation='relu')
        self.batch_norm_extract2c = BatchNormalization()

        self.conv2d_extract3a = Conv2D(
            96, (2, 2), padding='same', activation='relu')
        self.batch_norm_extract3a = BatchNormalization()
        self.conv2d_extract3b = Conv2D(96, (3,3), padding='valid', activation='relu')
        self.batch_norm_extract3b = BatchNormalization()
        self.dropout_extract3 = Dropout(0.35)
        self.conv2d_extract3c = Conv2D(96, (1, 1), activation='relu')
        self.batch_norm_extract3c = BatchNormalization()

        self.conv2d_extract5a = Conv2D(96, (3, 3), strides=(
            2, 2), padding='same', activation='relu')
        self.batch_norm_extract5a = BatchNormalization()
        self.conv2d_extract5b = Conv2D(96, (3,3), padding='valid', activation='relu')
        self.batch_norm_extract5b = BatchNormalization()
        self.dropout_extract5 = Dropout(0.35)
        self.conv2d_extract5c = Conv2D(96, (1, 1), activation='relu')
        self.batch_norm_extract5c = BatchNormalization()

        self.conv2d_extract8a = Conv2D(96, (3, 3), strides=(
            2, 2), padding='same', activation='relu')
        self.batch_norm_extract8a = BatchNormalization()
        self.conv2d_extract8b = Conv2D(96, (3,3), strides=(2,2), padding='valid', activation='relu')
        self.batch_norm_extract8b = BatchNormalization()
        self.dropout_extract8 = Dropout(0.35)
        self.conv2d_extract8c = Conv2D(96, (1, 1), activation='relu')
        self.batch_norm_extract8c = BatchNormalization()

        self.conv2d_extract12a = Conv2D(96, (5,5), strides=(2,2), activation='relu', padding='same')
        self.batch_norm_extract12a = BatchNormalization()
        self.conv2d_extract12b = Conv2D(
            96, (6, 6), strides=(3, 3), activation='relu')
        self.batch_norm_extract12b = BatchNormalization()
        self.dropout_extract12 = Dropout(0.35)
        self.conv2d_extract12c = Conv2D(96, (1, 1), activation='relu')
        self.batch_norm_extract12c = BatchNormalization()

        # we need better gradient flow
        self.conv2d_extract = Conv2D(96, (3, 3), activation='relu', padding='same')
        self.batch_norm_extract = BatchNormalization()
        self.dropout_classify2 = Dropout(0.35)
        self.dropout_classify3 = Dropout(0.35)
        self.dropout_classify5 = Dropout(0.35)
        self.dropout_classify8 = Dropout(0.35)
        self.dropout_classify12 = Dropout(0.35)
        # use resnet connection here to optimize gradient flow
        self.classifier1 = Conv2D(256, (1, 1), activation='relu')
        self.objectnesses = Conv2D(1,(1,1),activation='linear')
        self.classifier2 = Conv2D(64, (1,1), activation='relu')
        self.classifier3 = Conv2D(3, (1,1), activation='linear')
        self.batch_norm_classify1 = BatchNormalization()
        self.batch_norm_classify2 = BatchNormalization()

        self.dropout_regressor2 = Dropout(0.35)
        self.dropout_regressor3 = Dropout(0.35)
        self.dropout_regressor5 = Dropout(0.35)
        self.dropout_regressor8 = Dropout(0.35)
        self.dropout_regressor12 = Dropout(0.35)
        self.regressor1 = Conv2D(512,(1,1),activation='relu')
        self.regressor2 = Conv2D(128,(1,1),activation='relu')
        self.regressor3 = Conv2D(4,(1,1),activation='linear')
        self.batch_norm_regress1 = BatchNormalization()
        self.batch_norm_regress2 = BatchNormalization()

    def call(self, input_tensor, training=False):
        features = self.dropout_condense1(input_tensor)
        features = self.conv2d_condense1(features)
        features = self.batch_norm1(features)
        features = self.dropout_condense2(features)
        features = self.conv2d_condense2(features)
        features = self.batch_norm2(features)

        extract2 = self.conv2d_extract2a(features)
        extract2 = self.batch_norm_extract2a(extract2)
        extract2 = self.conv2d_extract2b(extract2)
        extract2 = self.batch_norm_extract2b(extract2)
        extract2 = self.dropout_extract2(extract2)
        extract2 = self.conv2d_extract2c(extract2)
        extract2 = self.batch_norm_extract2c(extract2)

        extract3 = self.conv2d_extract3a(features)
        extract3 = self.batch_norm_extract3a(extract3)
        extract3 = self.conv2d_extract3b(extract3)
        extract3 = self.batch_norm_extract3b(extract3)
        extract3 = self.dropout_extract3(extract3)
        extract3 = self.conv2d_extract3c(extract3)
        extract3 = self.batch_norm_extract3c(extract3)

        extract5 = self.conv2d_extract5a(features)
        extract5 = self.batch_norm_extract5a(extract5)
        extract5 = self.conv2d_extract5b(extract5)
        extract5 = self.batch_norm_extract5b(extract5)
        extract5 = self.dropout_extract5(extract5)
        extract5 = self.conv2d_extract5c(extract5)
        extract5 = self.batch_norm_extract5c(extract5)

        extract8 = self.conv2d_extract8a(features)
        extract8 = self.batch_norm_extract8a(extract8)
        extract8 = self.conv2d_extract8b(extract8)
        extract8 = self.batch_norm_extract8b(extract8)
        extract8 = self.dropout_extract8(extract8)
        extract8 = self.conv2d_extract8c(extract8)
        extract8 = self.batch_norm_extract8c(extract8)

        extract12 = self.conv2d_extract12a(features)
        extract12 = self.batch_norm_extract12a(extract12)
        extract12 = self.conv2d_extract12b(extract12)
        extract12 = self.batch_norm_extract12b(extract12)
        extract12 = self.dropout_extract12(extract12)
        extract12 = self.conv2d_extract12c(extract12)
        extract12 = self.batch_norm_extract12c(extract12)

        extract2F = self.conv2d_extract(extract2)
        extract2F = self.batch_norm_extract(extract2F)
        extract2F += extract2
        extract2C = self.classifier1(extract2F)
        extract2C = self.dropout_classify2(extract2C)
        extract2O = self.objectnesses(extract2C)
        extract2C = self.classifier2(extract2C)
        extract2C = self.batch_norm_classify2(extract2C)
        extract2C = self.classifier3(extract2C)
        extract2R = self.regressor1(extract2F)
        extract2R = self.batch_norm_regress1(extract2R)
        extract2R = self.dropout_regressor2(extract2R)
        extract2R = self.regressor2(extract2R)
        extract2R = self.batch_norm_regress2(extract2R)
        extract2R = self.regressor3(extract2R)
        extract2 = tf.concate([extract2O,extract2C,extract2R],axis=-1)

        extract3F = self.conv2d_extract(extract3)
        extract3F = self.batch_norm_extract(extract3F)
        extract3F += extract3
        extract3C = self.classifier1(extract3F)
        extract3C = self.dropout_classify3(extract3C)
        extract3O = self.objectnesses(extract3C)
        extract3C = self.classifier2(extract3C)
        extract3C = self.batch_norm_classify2(extract3C)
        extract3C = self.classifier3(extract3C)
        extract3R = self.regressor1(extract3F)
        extract3R = self.batch_norm_regress1(extract3R)
        extract3R = self.dropout_regressor3(extract3R)
        extract3R = self.regressor2(extract3R)
        extract3R = self.batch_norm_regress2(extract3R)
        extract3R = self.regressor3(extract3R)
        extract3 = tf.concate([extract3O,extract3C,extract3R],axis=-1)

        extract5F = self.conv2d_extract(extract5)
        extract5F = self.batch_norm_extract(extract5F)
        extract5F += extract5
        extract5C = self.classifier1(extract5F)
        extract5C = self.dropout_classify5(extract5C)
        extract5O = self.objectnesses(extract5C)
        extract5C = self.classifier2(extract5C)
        extract5C = self.batch_norm_classify2(extract5C)
        extract5C = self.classifier3(extract5C)
        extract5R = self.regressor1(extract5F)
        extract5R = self.batch_norm_regress1(extract5R)
        extract5R = self.dropout_regressor5(extract5R)
        extract5R = self.regressor2(extract5R)
        extract5R = self.batch_norm_regress2(extract5R)
        extract5R = self.regressor3(extract5R)
        extract5 = tf.concate([extract5O,extract5C,extract5R],axis=-1)

        extract8F = self.conv2d_extract(extract8)
        extract8F = self.batch_norm_extract(extract8F)
        extract8F += extract8
        extract8C = self.classifier1(extract8F)
        extract8C = self.dropout_classify8(extract8C)
        extract8O = self.objectnesses(extract8C)
        extract8C = self.classifier2(extract8C)
        extract8C = self.batch_norm_classify2(extract8C)
        extract8C = self.classifier3(extract8C)
        extract8R = self.regressor1(extract8F)
        extract8R = self.batch_norm_regress1(extract8R)
        extract8R = self.dropout_regressor8(extract8R)
        extract8R = self.regressor2(extract8R)
        extract8R = self.batch_norm_regress2(extract8R)
        extract8R = self.regressor3(extract8R)
        extract8 = tf.concate([extract8O,extract8C,extract8R],axis=-1)

        extract12F = self.conv2d_extract(extract12)
        extract12F = self.batch_norm_extract(extract12F)
        extract12F += extract12
        extract12C = self.classifier1(extract12F)
        extract12C = self.dropout_classify12(extract12C)
        extract12O = self.objectnesses(extract12C)
        extract12C = self.classifier2(extract12C)
        extract12C = self.batch_norm_classify2(extract12C)
        extract12C = self.classifier3(extract12C)
        extract12R = self.regressor1(extract12F)
        extract12R = self.batch_norm_regress1(extract12R)
        extract12R = self.dropout_regressor12(extract12R)
        extract12R = self.regressor2(extract12R)
        extract12R = self.batch_norm_regress2(extract12R)
        extract12R = self.regressor3(extract12R)
        extract12 = tf.concate([extract12O,extract12C,extract12R],axis=-1)
        # if we consider each pixel as an region proposal
        # there's at least 775 regions proposed for a single image
        return [extract2,extract3,extract5,extract8,extract12]

    def build_graph(self, input_shape):
        x = Input(shape=input_shape)
        return Model(inputs=[x], outputs=self.call(x))


class FullProposer(Model):
    def __init__(self, preprocessor):
        super(FullProposer, self).__init__()
        self.preprocessor = preprocessor
        self.preprocessor.trainable = False
        self.proposer = PixelRPN()
        self.block6_layer_counts = {'conv2d':10, 'batch_normalization':10, 'activation':10, 'average_pooling2d':1, 'mixed':1}

    def call(self, input_imgs, training=False):
        features = self.preprocessor(input_imgs)
        preds = self.proposer(features)
        return preds

    def set_layers_trainability(self, trainable, block,layer_counts):
        for name in layer_counts:
            if layer_counts[name]==1:
                if name=='mixed':
                    self.preprocessor.get_layer(f'{name}{block+1}').trainable=trainable
                else:
                    self.preprocessor.get_layer(f'{name}_{block}').trainable=trainable
            else:
                for i in range(layer_counts[name]):
                    self.preprocessor.get_layer(f'{name}_{block}{i}').trainable=trainable

# class RCNN(Model):
#     @tf.function(input_signature=[tf.TensorSpec(shape=(1,None,None,192), dtype=tf.float32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(), dtype=tf.int32),tf.TensorSpec(shape=(3), dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.int32)])
#     def getROIfeature(self, features, frow, fcol, inputs, size):
#         label = tf.cast(inputs[0],tf.int32)
#         rrow = tf.constant(0,dtype=tf.int32)
#         ccol = tf.constant(0,dtype=tf.int32)
#         if size==0:
#             rrow = calc_func[0](frow)
#             ccol = calc_func[0](fcol)
#         elif size==1:
#             rrow = calc_func[1](frow)
#             ccol = calc_func[1](fcol)
#         elif size==2:
#             rrow = calc_func[2](frow)
#             ccol = calc_func[2](fcol)
#         elif size==3:
#             rrow = calc_func[3](frow)
#             ccol = calc_func[3](fcol)
#         elif size==4:
#             rrow = calc_func[4](frow)
#             ccol = calc_func[4](fcol)
#         else:
#             tf.print("error")
#         r = tf.cast(tf.math.floor(tf.cast(inputs[1],tf.int32)*(frow-self.sizes[size])/(rrow-1)),tf.int32)
#         c = tf.cast(tf.math.floor(tf.cast(inputs[2],tf.int32)*(fcol-self.sizes[size])/(ccol-1)),tf.int32)
#         rEnd = r+self.sizes[size]
#         cEnd = c+self.sizes[size]
#         return features[label, r:rEnd, c:cEnd, :]

#     def __init__(self):
#         super(RCNN, self).__init__()
#         self.sizes = tf.constant(
#             [2, 3, 5, 8, 12], dtype=tf.int32)
#         # takes input of (17,17,768) or larger
#         self.conv2d_condense1 = Conv2D(
#             384, (1, 1), padding='same', activation='relu',name='conv2d_condense1')
#         self.batch_norm1 = BatchNormalization()
#         # if we set this number too small, it creates information bottleneck
#         # if it's high, we cannot use resnet connection
#         self.conv2d_condense2 = Conv2D(
#             192, (3, 3), padding='same', activation='relu',name='conv2d_condense2')
#         self.batch_norm2 = BatchNormalization()


#         self.conv2d_extract12_condense = Conv2D(96, (1, 1), activation='relu',name='conv2d_extract12_condense')
#         self.batch_norm_extract12a = BatchNormalization()
#         self.conv2d_extract12_supercondense = Conv2D(48, (1, 1), activation='relu',name='conv2d_extract12_supercondense')
#         self.batch_norm_extract12b = BatchNormalization()
#         self.conv2d_12_8 = Conv2D(48, (5,5), activation='relu',name='conv2d_12_8')
#         self.batch_norm_extract12c = BatchNormalization()
#         self.conv2d_extract12_5 = Conv2D(96,(4,4),strides=(2,2),activation='relu',name='conv2d_extract12_5')
#         self.batch_norm_extract12d = BatchNormalization()

#         self.conv2d_extract8_condense = Conv2D(96, (1, 1), activation='relu')
#         self.batch_norm_extract8a = BatchNormalization()
#         self.conv2d_extract8_supercondense = Conv2D(48, (1, 1), activation='relu')
#         self.batch_norm_extract8b = BatchNormalization()
#         self.conv2d_8_5 = Conv2D(48, (4,4), activation='relu')
#         self.batch_norm_extract8c = BatchNormalization()
#         self.conv2d_extract8_3 = Conv2D(96,(4,4),strides=(2,2),activation='relu')
#         self.batch_norm_extract8d = BatchNormalization()

#         self.conv2d_extract5_condense = Conv2D(96, (1, 1), activation='relu')
#         self.batch_norm_extract5a = BatchNormalization()
#         self.conv2d_extract5_supercondense = Conv2D(48, (1, 1), activation='relu')
#         self.batch_norm_extract5b = BatchNormalization()
#         self.conv2d_5_3 = Conv2D(48, (3,3), activation='relu')
#         self.batch_norm_extract5c = BatchNormalization()
#         self.conv2d_extract5_2 = Conv2D(96,(3,3),strides=(2,2),activation='relu')
#         self.batch_norm_extract5d = BatchNormalization()

#         self.conv2d_extract3_condense = Conv2D(96, (1, 1), activation='relu')
#         self.batch_norm_extract3a = BatchNormalization()
#         self.conv2d_extract3_supercondense = Conv2D(48, (1, 1), activation='relu')
#         self.batch_norm_extract3b = BatchNormalization()
#         self.conv2d_3_2 = Conv2D(48, (2,2), activation='relu')
#         self.batch_norm_extract3c = BatchNormalization()
#         self.conv2d_extract3_2 = Conv2D(96,(2,2),strides=(1,1),activation='relu')
#         self.batch_norm_extract3d = BatchNormalization()

#         self.conv2d_extract2_condense = Conv2D(144, (1, 1), activation='relu')
#         self.batch_norm_extract2a = BatchNormalization()
#         self.conv2d_extract2 = Conv2D(256, (2, 2), activation='relu')
#         self.batch_norm_extract2b = BatchNormalization()

#         self.flatten = Flatten()

#         self.classifier1 = Dense(256, activation='relu')
#         self.classifier2 = Dense(64, activation='relu')
#         self.classifier3 = Dense(3, activation='softmax')

#         self.regressor1 = Dense(256, activation='relu')
#         self.regressor2 = Dense(256, activation='relu')
#         self.regressor3 = Dense(4, activation='linear')

#     def call(self, input_features, pixel_objectness, training=False):
#         extract2, extract3, extract5, extract8, extract12 = pixel_objectness
#         # ignore extract0 for now
#         self.features = self.conv2d_condense1(input_features)
#         self.features = self.batch_norm1(self.features)
#         self.features = self.conv2d_condense2(self.features)
#         self.features = self.batch_norm2(self.features)
#         self.frow = tf.shape(self.features)[1]
#         self.fcol = tf.shape(self.features)[2]
#         # extract2
#         filtered2 = tf.where(extract2 > 0.5)
#         shape2 = (2, 2, 192)
#         if filtered2.shape[0] == 0:
#             features2 = tf.zeros((0, *shape2))
#         else:
#             features2 = tf.map_fn(lambda pos : self.getROIfeature(self.features,self.frow,self.fcol,pos, 0), filtered2, fn_output_signature=tf.TensorSpec(shape=shape2, dtype=tf.float32))
#         # expected shape (new_batch_size,3,3,#channels)

#         features2 = self.conv2d_extract2_condense(features2)
#         features2 = self.batch_norm_extract2a(features2)
#         features2 = self.conv2d_extract2(features2)
#         features2 = self.batch_norm_extract2b(features2)

#         filtered3 = tf.where(extract3 > 0.5)
#         shape3 = (3, 3, 192)
#         if filtered3.shape[0] == 0:
#             features3 = tf.zeros((0, *shape3))
#         else:
#             features3 = tf.map_fn(lambda pos : self.getROIfeature(self.features,self.frow,self.fcol,pos, 1), filtered3, fn_output_signature=tf.TensorSpec(shape=shape3, dtype=tf.float32))

#         features3 = self.conv2d_extract3_condense(features3)
#         features3 = self.batch_norm_extract3a(features3)
#         features3Condensed = self.conv2d_extract3_supercondense(features3)
#         features3Condensed = self.batch_norm_extract3b(features3Condensed)
#         features3Condensed = self.conv2d_3_2(features3Condensed)
#         features3Condensed = self.batch_norm_extract3c(features3Condensed)
#         features3 = self.conv2d_extract3_2(features3)
#         features3 = self.batch_norm_extract3d(features3)
#         features3 = tf.concat([features3Condensed, features3], axis=-1)
#         features3 = self.conv2d_extract2(features3)
#         features3 = self.batch_norm_extract2b(features3)

#         filtered5 = tf.where(extract5 > 0.5)
#         shape5 = (5, 5, 192)
#         if filtered5.shape[0] == 0:
#             features5 = tf.zeros((0, *shape5))
#         else:
#             features5 = tf.map_fn(lambda pos : self.getROIfeature(self.features,self.frow,self.fcol,pos, 2), filtered5, fn_output_signature=tf.TensorSpec(shape=shape5, dtype=tf.float32))

#         features5 = self.conv2d_extract5_condense(features5)
#         features5 = self.batch_norm_extract5a(features5)
#         features5Condensed = self.conv2d_extract5_supercondense(features5)
#         features5Condensed = self.batch_norm_extract5b(features5Condensed)
#         features5Condensed = self.conv2d_5_3(features5Condensed)
#         features5Condensed = self.batch_norm_extract5c(features5Condensed)
#         features5Condensed = self.conv2d_3_2(features5Condensed)
#         features5Condensed = self.batch_norm_extract3c(features5Condensed)
#         features5 = self.conv2d_extract5_2(features5)
#         features5 = self.batch_norm_extract5d(features5)
#         features5 = tf.concat([features5Condensed, features5], axis=-1)
#         features5 = self.conv2d_extract2(features5)
#         features5 = self.batch_norm_extract2b(features5)

#         filtered8 = tf.where(extract8 > 0.5)
#         shape8 = (8, 8, 192)
#         if filtered8.shape[0] == 0:
#             features8 = tf.zeros((0, *shape8))
#         else:
#             features8 = tf.map_fn(lambda pos : self.getROIfeature(self.features,self.frow,self.fcol,pos, 3), filtered8, fn_output_signature=tf.TensorSpec(shape=shape8, dtype=tf.float32))

#         features8 = self.conv2d_extract8_condense(features8)
#         features8 = self.batch_norm_extract8a(features8)
#         features8Condensed = self.conv2d_extract8_supercondense(features8)
#         features8Condensed = self.batch_norm_extract8b(features8Condensed)
#         features8Condensed = self.conv2d_8_5(features8Condensed)
#         features8Condensed = self.batch_norm_extract8c(features8Condensed)
#         features8Condensed = self.conv2d_5_3(features8Condensed)
#         features8Condensed = self.batch_norm_extract5c(features8Condensed)
#         features8Condensed = self.conv2d_3_2(features8Condensed)
#         features8Condensed = self.batch_norm_extract3c(features8Condensed)
#         features8 = self.conv2d_extract8_3(features8)
#         features8 = self.batch_norm_extract8d(features8)
#         features8 = self.conv2d_extract3_2(features8)
#         features8 = self.batch_norm_extract3d(features8)
#         features8 = tf.concat([features8Condensed, features8], axis=-1)
#         features8 = self.conv2d_extract2(features8)
#         features8 = self.batch_norm_extract2b(features8)

#         filtered12 = tf.where(extract12 > 0.5)
#         shape12 = (12, 12, 192)
#         if filtered12.shape[0] == 0:
#             features12 = tf.zeros((0, *shape12))
#         else:
#             features12 = tf.map_fn(lambda pos : self.getROIfeature(self.features,self.frow,self.fcol,pos, 4), filtered12, fn_output_signature=tf.TensorSpec(shape=shape12, dtype=tf.float32))

#         features12 = self.conv2d_extract12_condense(features12)
#         features12 = self.batch_norm_extract12a(features12)
#         features12Condensed = self.conv2d_extract12_supercondense(features12)
#         features12Condensed = self.batch_norm_extract12b(features12Condensed)
#         features12Condensed = self.conv2d_12_8(features12Condensed)
#         features12Condensed = self.batch_norm_extract12c(features12Condensed)
#         features12Condensed = self.conv2d_8_5(features12Condensed)
#         features12Condensed = self.batch_norm_extract8c(features12Condensed)
#         features12Condensed = self.conv2d_5_3(features12Condensed)
#         features12Condensed = self.batch_norm_extract5c(features12Condensed)
#         features12Condensed = self.conv2d_3_2(features12Condensed)
#         features12Condensed = self.batch_norm_extract3c(features12Condensed)
#         features12 = self.conv2d_extract12_5(features12)
#         features12 = self.batch_norm_extract12d(features12)
#         features12 = self.conv2d_extract5_2(features12)
#         features12 = self.batch_norm_extract5d(features12)
#         features12 = tf.concat([features12Condensed, features12], axis=-1)
#         features12 = self.conv2d_extract2(features12)
#         features12 = self.batch_norm_extract2b(features12)

#         features2 = self.flatten(features2)
#         features3 = self.flatten(features3)
#         features5 = self.flatten(features5)
#         features8 = self.flatten(features8)
#         features12 = self.flatten(features12)

#         class2 = self.classifier1(features2)
#         class3 = self.classifier1(features3)
#         class5 = self.classifier1(features5)
#         class8 = self.classifier1(features8)
#         class12 = self.classifier1(features12)

#         class2 = self.classifier2(class2)
#         class3 = self.classifier2(class3)
#         class5 = self.classifier2(class5)
#         class8 = self.classifier2(class8)
#         class12 = self.classifier2(class12)

#         class2 = self.classifier3(class2)
#         class3 = self.classifier3(class3)
#         class5 = self.classifier3(class5)
#         class8 = self.classifier3(class8)
#         class12 = self.classifier3(class12)

#         regress2 = self.regressor1(features2)
#         regress3 = self.regressor1(features3)
#         regress5 = self.regressor1(features5)
#         regress8 = self.regressor1(features8)
#         regress12 = self.regressor1(features12)

#         regress2 = self.regressor2(regress2)
#         regress3 = self.regressor2(regress3)
#         regress5 = self.regressor2(regress5)
#         regress8 = self.regressor2(regress8)
#         regress12 = self.regressor2(regress12)

#         regress2 = self.regressor3(regress2)
#         regress3 = self.regressor3(regress3)
#         regress5 = self.regressor3(regress5)
#         regress8 = self.regressor3(regress8)
#         regress12 = self.regressor3(regress12)
#         # post processing using position information and returning
#         # try optimize gradient flow
#         return [[class2, regress2, filtered2], [class3, regress3, filtered3], [class5, regress5, filtered5], [class8, regress8, filtered8], [class12, regress12, filtered12]]

#     def build_graph(self, input_shape):
#         features = Input(shape=input_shape)
#         objectness = [Input(shape=(16, 16)), Input(shape=(15, 15)), Input(
#             shape=(7, 7)), Input(shape=(4, 4)), Input(shape=(2, 2))]
#         return Model(inputs=[features, objectness], outputs=self(features, objectness))


# class FullRCNN(Model):
#     def __init__(self, preprocessor):
#         super(FullRCNN, self).__init__()
#         self.preprocessor = preprocessor
#         self.preprocessor.trainable = False
#         self.RCNN = RCNN()
#         self.block6_layer_counts = {'conv2d':10, 'batch_normalization':10, 'activation':10, 'average_pooling2d':1, 'mixed':1}

#     def call(self, input_imgs, input_objs, training=False):
#         features = self.preprocessor(input_imgs)
#         ret = self.RCNN(features, input_objs, training)
#         return ret

#     def set_layers_trainability(self, trainable, block,layer_counts):
#         for name in layer_counts:
#             if layer_counts[name]==1:
#                 if name=='mixed':
#                     self.preprocessor.get_layer(f'{name}{block+1}').trainable=trainable
#                 else:
#                     self.preprocessor.get_layer(f'{name}_{block}').trainable=trainable
#             else:
#                 for i in range(layer_counts[name]):
#                     self.preprocessor.get_layer(f'{name}_{block}{i}').trainable=trainable

ProposerModel = FullProposer(preprocessor)

# FullRCNNModel = FullRCNN(preprocessor)