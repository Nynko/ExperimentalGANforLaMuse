import os
import time
import tensorflow as tf
import numpy as np
import random
import re
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from GAN_functions_class import GanFunctionsClass


## Variables globales
channels = 3
img_shape = (256,256,channels)

path_img= 'Images/'
path_Collage = "Images/Collage/"
path_Real = 'Images/Real/'

LAMBDA = 4


class GanFunctionsDiscriminateurDePix2Pix(GanFunctionsClass):

# =================================================================================== #
#               4. Define the discriminator and generator losses                      #
# =================================================================================== # 

    def build_dicriminator(self,image_shape):
        # weight initialization
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        # source image input
        in_src_image = tf.keras.layers.Input(shape=image_shape)
        # target image input
        in_target_image = tf.keras.layers.Input(shape=image_shape)
        # concatenate images channel-wise
        merged = tf.keras.layers.Concatenate()([in_src_image, in_target_image])
        # C64
        d = tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        # C128
        d = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = tf.keras.layers.BatchNormalization()(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        # C256
        d = tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = tf.keras.layers.BatchNormalization()(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        # C512
        d = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = tf.keras.layers.BatchNormalization()(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = tf.keras.layers.BatchNormalization()(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        # patch output
        d = tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = tf.keras.layers.Activation('sigmoid')(d)
        # define model
        model = tf.keras.models.Model([in_src_image, in_target_image], patch_out)

        return model
