## 
# https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://github.com/eriklindernoren/Keras-GAN/blob/master/context_encoder/context_encoder.py



from calendar import c
import os
from sys import displayhook
import time
import tensorflow as tf
import numpy as np
import random

from OLD2.CreateNPY import *

from keras.preprocessing.image import array_to_img


## Variables globales

channels = 3
img_shape = (256,256,channels)

BATCH_SIZE = 50
LAMBDA = 0.1


path_img= 'Images/'
path_Collage = "Images/Collage/"
path_Real = 'Images/Real/'



# =================================================================================== #
#               4. Define the discriminator and generator losses                      #
# =================================================================================== # 


# def wasserstein_loss(y_true, y_pred):
#     return tf.keras.K.mean(y_true * y_pred)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output,disc_fake_output, disc_generated_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(disc_fake_output), disc_fake_output)
  generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = fake_loss + generated_loss + real_loss

  return total_disc_loss


def generator_loss(disc_generated_output, gen_output, original):
  gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.norm(tf.abs(tf.cast(original,tf.float32) - tf.cast(gen_output,tf.float32)))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


# =================================================================================== #
#               6. Define the  generator                                              #
# =================================================================================== #  

def build_generator(img_shape):

    model = tf.keras.Sequential()

    # Encoder
    model.add( tf.keras.layers.Input(shape=img_shape))
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

# =================================================================================== #
#               7. Define the discriminator                                           #
# =================================================================================== # 

def build_discriminator(img_shape):
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

    model.summary()

    return model

    
## UTILS

def getTupleRandom(batchsize):
    listCollages = os.listdir("./Images/Collage")
    listReal = os.listdir("./Images/Real")
    random.shuffle(listCollages)
    random.shuffle(listReal)

    minGlobal = min(len(listCollages), len(listReal))
    compt_batch = 0
    listTuple = []
   
    while compt_batch*batchsize < minGlobal :
        tuple = (listReal[compt_batch*batchsize:(compt_batch+1)*batchsize],listCollages[compt_batch*batchsize:(compt_batch+1)*batchsize])
        listTuple.append(tuple)
        compt_batch += 1

    return listTuple


def LoadImages(batch,batchsize):

    # Check if the folders exists
    if not os.path.exists(path_img):
        print("Images folder not found")
        exit(1)
    if not os.path.exists(path_Collage):
        print("Collage not found")
        exit(1)
    if not os.path.exists(path_Real):
        print("Real not found")
        exit(1)

    # Create the tensor for the images of size (batchsize,height,width,channels)
    images_real = np.zeros((batchsize,img_shape[0],img_shape[1],img_shape[2]))
    images_collage = np.zeros((batchsize,img_shape[0],img_shape[1],img_shape[2]))

    #  Load all the images in the folder Images/Collage
    images_collage = []
    images_real = []
    real = batch[0]
    fake = batch[1]
    for img_real in real:
        img_real = load_img(path_Real + img_real)
        #Resize the images
        img_real = img_to_array(img_real)
        # Resize and normalize the images between -1 and 1
        img_real = tf.image.resize_with_pad(img_real,img_shape[0],img_shape[1])/127.5 -1
        # Add the images to the tensor
        images_real.append(img_real)

    for img_fake in fake:
        img_fake = load_img(path_Collage + img_fake)
        #Resize the images
        img_fake = img_to_array(img_fake)
        # Resize and normalize the images between -1 and 1
        img_fake = tf.image.resize_with_pad(img_fake,img_shape[0],img_shape[1])/127.5 -1
        # Add the images to the tensor
        images_collage.append(img_fake)
        
    images_collage = np.array(images_collage,dtype=object).astype('float64')
    images_real = np.array(images_real,dtype=object).astype('float64')
    
    return (images_real,images_collage)
    



# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(fake_images, real_images):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(fake_images, training=True)

        real_output = discriminator(real_images, training=True)
        fake_generated_output = discriminator(generated_images, training=True)
        fake_output2 = discriminator(fake_images, training=True)

        disc_loss = discriminator_loss(real_output,fake_output2, fake_generated_output)
        gen_loss = generator_loss(fake_generated_output,generated_images,real_images)


        # loss_euclidienne = tf.norm(tf.cast(generated_images,tf.float32) - tf.cast(real_images,tf.float32),ord="euclidean") # Calcul de la distance euclidienne
        # gen_loss = gen_loss + loss_euclidienne

        # if(disc_loss1 < disc_loss2): # Si le générateur rend un peu meilleur l'image
        #     disc_loss = disc_loss1
        #     gen_loss = gen_loss*0.9
        # else:
        #     disc_loss = disc_loss1

        # disc_loss += loss_euclidienne

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(listeTuple,batch_size,step,epochs):
    if epochs > len(listeTuple):
        print("Epochs too high")
        exit(1)

    step2 = step.numpy().item()
    epoch = step2 

    while epoch <= epochs:
        start = time.time()

        batch = liste_Tuple[epoch]
        batch_real, batch_fake = LoadImages(batch,batch_size)
        gen_loss, disc_loss = train_step(batch_real,batch_fake)

        # Print model losses
        print("")
        print("Generator loss: Total: {}  - gan: {} - distance: {}".format(gen_loss[0],gen_loss[1],gen_loss[2]))
        print("Discriminator loss: {}".format(disc_loss))


        # Save the model every 10 epochs
        step.assign(epoch)
        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
        epoch += 1


# def showImage(img):
#     img = np.add(img,1)
#     img = array_to_img(img, scale=False)
#     img.show()




if __name__ == "__main__":

    ## Batch
    # Create batches of fake images and real images
    # We will use the same batch size for both the real and fake images
    batch_size = BATCH_SIZE
    # Create the batches
    # real_images_batch = tf.data.Dataset.from_tensor_slices(real_images).batch(batch_size)
    # collage_images_batch = tf.data.Dataset.from_tensor_slices(collage_images).batch(batch_size)

    # Create the batches
    liste_Tuple = getTupleRandom(batch_size)


    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


    # =================================================================================== #
    #                             1. Build the generator                                  #
    # =================================================================================== #
    generator = build_generator(img_shape)
    generator.compile(loss=generator_loss, optimizer=generator_optimizer)

    ## Test d'une image sur le generateur
    # nb = 1
    # img = images[nb].reshape(1,1024,1024,3)
    #Si des fichiers de checkpoints existent, on les charge
    # generated_image = generator(img, training=False)
    # img = array_to_img(generated_image[0], scale=False)
    # img.show()
    # image = array_to_img(images[nb], scale=False)
    # image.show()


    # =================================================================================== #
    #                             2. Build and compile the discriminator                  #
    # =================================================================================== #

    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss=[discriminator_loss],
                                optimizer=discriminator_optimizer,
                                metrics=['accuracy'])

    # decision = discriminator(generated_image)
    # print (decision)     



    # =================================================================================== #
    #               3. The combined model (stacked generator and discriminator)           #
    #               Trains the generator to fool the discriminator                        #
    # =================================================================================== #         

    ### DCGAN 
    # gan = tf.keras.Sequential()
    # gan.add(generator)
    # gan.add(discriminator)
    # discriminator.trainable = False 
    # gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['mae'])


    step=tf.Variable(1)
    checkpoint_dir = "./Checkpoints/Inpainting2/"
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
        print("Weights loaded")
    else:
        # generator.load_weights('{}/weight_{}.h5'.format("./Checkpoints",99))
        print("Checkpoint not found")


    train(liste_Tuple,batch_size, step, epochs=200)

