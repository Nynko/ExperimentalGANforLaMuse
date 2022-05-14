import os
import time
import tensorflow as tf
import numpy as np
import random
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

LAMBDA = 0.05


class GanFunctionsInpainting(GanFunctionsClass):

# =================================================================================== #
#               6. Define the  generator                                              #
# =================================================================================== #  

 def generator_loss(self,output_discriminateur, generated, collage_original): 
        loss = self.cross_entropy(tf.ones_like(output_discriminateur), output_discriminateur)

        # Mean absolute error
        # l1_loss = tf.reduce_mean(tf.abs(tf.cast(collage_original,tf.float32) - tf.cast(generated,tf.float32)))
        
        # Norme euclidienne
        l1_loss = tf.norm(tf.cast(collage_original,tf.float32) - tf.cast(generated,tf.float32),ord='euclidean')


        total_gen_loss = loss + (LAMBDA * l1_loss)
    

        return total_gen_loss

# https://github.com/eriklindernoren/Keras-GAN/blob/master/context_encoder/context_encoder.py

def build_generator(self,img_shape):

    model = tf.keras.Sequential()

    # Encoder
    model.add(tf.keras.layers.Input(shape=img_shape))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Conv2D(512, kernel_size=1, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.5))

    # Decoder

    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation('tanh'))

    model.summary()

    masked_img = tf.keras.layers.Input(shape=img_shape)
    gen_missing = model(masked_img)

    return tf.keras.Model(masked_img, gen_missing)
