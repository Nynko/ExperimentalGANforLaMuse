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
path_img= 'Images/'
path_Collage = "Images/Collage/"
path_Real = 'Images/Real/'

LAMBDA = 1


class GanFunctionsInpainting(GanFunctionsClass):

# =================================================================================== #
#               4. Define the discriminator and generator losses                      #
# =================================================================================== # 

    # def discriminator_loss(real_output, fake_output):
    #     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    #     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #     total_loss = real_loss + fake_loss
    #     return total_loss

    def discriminator_loss(self,disc_real_output, disc_generated_output):
        real_loss = self.cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss


    def generator_loss(self,output_discriminateur): 
        loss = self.cross_entropy(tf.ones_like(output_discriminateur), output_discriminateur)
        return loss


    # def generator_loss(disc_generated_output, gen_output, original):
    #   gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    #   # Mean absolute error
    #   l1_loss = tf.reduce_mean(tf.abs(tf.cast(original,tf.float32) - tf.cast(gen_output,tf.float32)))

    #   total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    #   return total_gen_loss #, gan_loss, l1_loss



# =================================================================================== #
#               6. Define the  generator                                              #
# =================================================================================== #  

    def build_generator(self,img_shape):
        gf = 64
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=img_shape),
                tf.keras.layers.Conv2D(gf,(5, 5), dilation_rate=2, input_shape=img_shape, padding="same",name="enc_conv_1"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(gf,(5, 5), dilation_rate=2, padding="same",name="enc_conv_2"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(gf*2,(5, 5), dilation_rate=2, padding="same",name="enc_conv_3"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(gf*4,(5, 5), dilation_rate=2, padding="same",name="enc_conv_4"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(gf*8,(5, 5), dilation_rate=2, padding="same",name="enc_conv_5"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dropout(0.5),

                # Decoder
                tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
                tf.keras.layers.Conv2DTranspose(gf*8,(3, 3), dilation_rate=2, padding="same",name="upsample_conv_1"),
                tf.keras.layers.Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT')),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
                tf.keras.layers.Conv2DTranspose(gf*4,(3, 3), dilation_rate=2, padding="same",name="upsample_conv_2"),
                tf.keras.layers.Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT')),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
                tf.keras.layers.Conv2DTranspose(gf*2,(3, 3), dilation_rate=2, padding="same",name="upsample_conv_3"),
                tf.keras.layers.Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT')),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
                tf.keras.layers.Conv2DTranspose(gf,(3, 3), dilation_rate=2, padding="same",name="upsample_conv_4"),
                tf.keras.layers.Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT')),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.BatchNormalization(momentum=0.8),

                tf.keras.layers.Conv2DTranspose(self.img_shape[2],(3, 3), dilation_rate=2, padding="same",name="final_output"),
                tf.keras.layers.Activation('tanh'),
            ]
        )
        return model 

# =================================================================================== #
#               7. Define the discriminator                                           #
# =================================================================================== # 


    def build_discriminator(self,img_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=img_shape)),
        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model



    # def build_discriminator(img_shape):
    #     model = tf.keras.Sequential(
    #         [
    #             tf.keras.Input(shape=img_shape),
    #             tf.keras.layers.Conv2D(df,kernel_size=5,strides=2,padding="same",name="disc_conv_1"),
    #             tf.keras.layers.LeakyReLU(alpha=0.2),
    #             tf.keras.layers.Dropout(0.25),

    #             tf.keras.layers.Conv2D(df*2,kernel_size=3,strides=2,padding="same",name="disc_conv_2"),
    #             tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1))),
    #             tf.keras.layers.BatchNormalization(momentum=0.8),
    #             tf.keras.layers.LeakyReLU(alpha=0.2),
    #             tf.keras.layers.Dropout(0.25),

    #             tf.keras.layers.Conv2D(df*4,kernel_size=3,strides=2,padding="same",name="disc_conv_3"),
    #             tf.keras.layers.BatchNormalization(momentum=0.8),
    #             tf.keras.layers.LeakyReLU(alpha=0.2),
    #             tf.keras.layers.Dropout(0.25),

    #             tf.keras.layers.Conv2D(df*8,kernel_size=3,strides=2,padding="same",name="disc_conv_4"),
    #             tf.keras.layers.BatchNormalization(momentum=0.8),
    #             tf.keras.layers.LeakyReLU(alpha=0.2),
    #             tf.keras.layers.Dropout(0.25),
    #             tf.keras.layers.Flatten(),
    #             tf.keras.layers.Dense(1)
    #         ]
    #     )
    #     print("====Discriminator Summary===")
    #     model.summary()
    #     return model   

