## 
# https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391
# https://www.tensorflow.org/tutorials/generative/dcgan
# Pix2Pix


import os
import tensorflow as tf
import numpy as np
from GAN_functions_class import GanFunctionsClass
from GAN_functions_inpainting1 import GanFunctionsInpainting
from GAN_functions_DiscriminateurDePix2Pix import LAMBDA, GanFunctionsDiscriminateurDePix2Pix



## Variables globales

BATCH_SIZE = 100
IMG_SHAPE = (256, 256, 3)
EPOCHS= 350

LAMBDA = 0


if __name__ == "__main__":


    # Create a GanFunctionsClass object
    gan = GanFunctionsClass(BATCH_SIZE,IMG_SHAPE,LAMBDA)

    ## Batch
    batch_size = BATCH_SIZE
    # Create the batches
    liste_Tuple = gan.getTupleRandom(batch_size)
    # --> [ (100imagesReal,100imagesCollage), (100imagesReal,100imagesCollage), ... ]

    img_shape= IMG_SHAPE

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


    # =================================================================================== #
    #                             1. Build the generator                                  #
    # =================================================================================== #
    generator = gan.build_generator(img_shape)
    generator.compile(loss=gan.generator_loss, optimizer=generator_optimizer)
    print(generator.summary())


    # =================================================================================== #
    #                             2. Build and compile the discriminator                  #
    # =================================================================================== #

    discriminator = gan.build_discriminator(img_shape)
    discriminator.compile(loss=[gan.discriminator_loss],
                                optimizer=discriminator_optimizer,
                                metrics=['accuracy'])
    print(discriminator.summary()) 


    # =================================================================================== #
    #               3. Define Checkpoints and the list of losses                          # 
    # =================================================================================== #         


    loss_liste= []
    step=tf.Variable(1)
    checkpoint_dir = "./Checkpoints/"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator,
                                    step=step)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)


    if(latest != None):
        print("Checkpoint found")
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        loss_liste = np.load(checkpoint_dir +"loss_liste.npy")
        loss_liste = loss_liste.tolist()
        print("Weights loaded")
    else:
        # generator.load_weights('{}/weight_{}.h5'.format("./Checkpoints",99))
        print("Checkpoint not found")

    epochs= EPOCHS

    gan.train(checkpoint, checkpoint_prefix,checkpoint_dir,generator, discriminator,generator_optimizer, discriminator_optimizer, liste_Tuple,batch_size, step, loss_liste, epochs)
    # print(loss_liste)
    # gan.showLosses(loss_liste)


