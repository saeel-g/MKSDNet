import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import backend

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.models import Model


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + "_block" + str(i + 1))
    return x
def transition_block(x, reduction, name):
    bn_axis = -1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_bn")(x)
    x = Activation("relu", name=name + "_relu")(x)
    x = Conv2D(int(backend.int_shape(x)[bn_axis] * reduction),1,use_bias=False,name=name + "_conv",)(x)
    x = AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x
def conv_block(x, growth_rate, name):
    bn_axis = -1 
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(x)
    x1 = Activation("relu", name=name + "_0_relu")(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + "_1_conv")(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x1)
    x1 = Activation("relu", name=name + "_1_relu")(x1)
    x1 = Conv2D(growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv")(x1)
    x = Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x

def DenseNet(blocks,include_top=True,input_tensor=None,input_shape=None,pooling=None,classes=2,classifier_activation="sigmoid"):
    img_input = Input(input_shape)
    bn_axis = -1
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name="conv1/conv")(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1/bn")(x)
    x = Activation("relu", name="conv1/relu")(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name="pool1")(x)

    x = dense_block(x, blocks[0], name="conv2")
    x = transition_block(x, 0.5, name="pool2")
    x = dense_block(x, blocks[1], name="conv3")
    x = transition_block(x, 0.5, name="pool3")
    x = dense_block(x, blocks[2], name="conv4")
    x = transition_block(x, 0.5, name="pool4")
    x = dense_block(x, blocks[3], name="conv5")
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = Activation("relu", name="relu")(x)

    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Flatten()(x)

    x = Dense(classes, activation=classifier_activation, name="predictions")(x)
           
    model = Model(img_input, x, name="densenet121")
    return model

########################################################################################################
def dense_block2(x, blocks, name):
    for i in range(blocks):
        x = conv_block2(x, 32, name=name + "_block" + str(i + 1))
    return x

def transition_block2(x, reduction, name):
    bn_axis = -1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_bn")(x)
    x = Activation("relu", name=name + "_relu")(x)
    xs=x
    x = Conv2D(int(backend.int_shape(x)[bn_axis] * reduction),1,use_bias=False,name=name + "_conv",)(x)
    
    x = AveragePooling2D(2, strides=2, name=name + "_pool")(x)

    x1 = Conv2D(int(backend.int_shape(xs)[bn_axis] * reduction),3, strides=2, padding='same', use_bias=False, name=name + "_2_conv")(xs)

    x2 = Conv2D(int(backend.int_shape(xs)[bn_axis] * reduction), 3, dilation_rate=3,  padding='same',use_bias=False, name=name + "_3_conv")(xs)
    x2 = MaxPooling2D(2, strides=2)(x2)
    x = Concatenate(axis=bn_axis, name=name + "_concat")([x, x1, x2])




    return x
def conv_block2(x, growth_rate, name):
    bn_axis = -1 
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(x)
    x1 = Activation("relu", name=name + "_0_relu")(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + "_1_conv")(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x1)
    x1 = Activation("relu", name=name + "_1_relu")(x1)
    x1 = Conv2D(growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv")(x1)
    x = Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x

def DenseNet2(blocks,include_top=True,input_tensor=None,input_shape=None,pooling=None,classes=2,classifier_activation="sigmoid"):
    img_input = Input(input_shape)
    bn_axis = -1
    #x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(32, 7, strides=2, padding='same',use_bias=False, name="conv1/conv")(img_input)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1/bn")(x)
    x = Activation("relu", name="conv1/relu")(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name="pool1")(x)

    x = dense_block2(x, blocks[0], name="conv2")
    x = transition_block2(x, 0.5, name="pool2")
    x = dense_block2(x, blocks[1], name="conv3")
    x = transition_block2(x, 0.5, name="pool3")
    x = dense_block2(x, blocks[2], name="conv4")
    x = transition_block2(x, 0.5, name="pool4")
    x = dense_block2(x, blocks[3], name="conv5")
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = Activation("relu", name="relu")(x)

    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Flatten()(x)

    x = Dense(classes, activation=classifier_activation, name="predictions")(x)
           
    model = Model(img_input, x, name="densenet121")
    return model


#model= DenseNet(blocks=[6, 12, 24, 16],input_shape=(224,224,3))
#model.summary()