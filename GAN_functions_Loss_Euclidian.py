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

LAMBDA = 1


class GanFunctionsLossEuclidian(GanFunctionsClass):

# =================================================================================== #
#               4. Define the discriminator and generator losses                      #
# =================================================================================== # 

    def generator_loss(self,output_discriminateur, generated, collage_original): 
        loss = self.cross_entropy(tf.ones_like(output_discriminateur), output_discriminateur)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(tf.cast(collage_original,tf.float32) - tf.cast(generated,tf.float32)))

        total_gen_loss = loss + (LAMBDA * l1_loss)

        return total_gen_loss


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self,generator, discriminator,generator_optimizer, discriminator_optimizer,fake_images, real_images):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(fake_images, training=True)

            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(generated_images, training=True)
            fake_output2 = discriminator(fake_images, training=True)

            disc_loss = self.discriminator_loss(real_output, fake_output)
            disc_loss2 = self.discriminator_loss(real_output, fake_output2)
            disc_loss = disc_loss + disc_loss2

            gen_loss = self.generator_loss(fake_output,generated_images,fake_images)


        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss
