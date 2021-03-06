## 
# https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391
# https://www.tensorflow.org/tutorials/generative/dcgan
# Pix2Pix



from calendar import c
import dis
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

BATCH_SIZE = 200
LAMBDA = 0.5


path_img= 'Images/'
path_Collage = "Images/Collage/"
path_Real = 'Images/Real/'



# =================================================================================== #
#               4. Define the discriminator and generator losses                      #
# =================================================================================== # 


# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss

def discriminator_loss(real_output,disc_fake_output, disc_generated_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(disc_fake_output), disc_fake_output)
  generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = fake_loss + generated_loss + real_loss

  return total_disc_loss


# def generator_loss(output_discriminateur): 
#     loss = cross_entropy(tf.ones_like(output_discriminateur), output_discriminateur)
#     return loss

def generator_loss(disc_generated_output, gen_output, original):
  gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(tf.cast(original,tf.float32) - tf.cast(gen_output,tf.float32)))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss #, gan_loss, l1_loss

# =================================================================================== #
#               6. Define the  generator                                              #
# =================================================================================== #  


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

  
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	# add downsampling layer
	g = tf.keras.layers.Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = tf.keras.layers.BatchNormalization()(g, training=True)
	# leaky relu activation
	g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	# add upsampling layer
	g = tf.keras.layers.Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = tf.keras.layers.BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = tf.keras.layers.Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = tf.keras.layers.Concatenate()([g, skip_in])
	# relu activation
	g = tf.keras.layers.Activation('relu')(g)
	return g


def build_generator(image_shape):
	# weight initialization
	init = tf.keras.initializers.RandomNormal(stddev=0.02)
	# image input
	in_image = tf.keras.layers.Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = tf.keras.layers.Activation('relu')(b)
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = tf.keras.layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = tf.keras.layers.Activation('tanh')(g)
	# define model
	model = tf.keras.Model(in_image, out_image)
	return model

# =================================================================================== #
#               7. Define the discriminator                                           #
# =================================================================================== # 
  

# def build_discriminator(img_shape):
#   initializer = tf.random_normal_initializer(0., 0.02)

#   inp = tf.keras.layers.Input(shape=img_shape, name='input_image')

#   down1 = downsample(64, 4, False)(inp)  # (batch_size, 128, 128, 64)
#   down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
#   down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

#   zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
#   conv = tf.keras.layers.Conv2D(512, 4, strides=1,
#                                 kernel_initializer=initializer,
#                                 use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

#   batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

#   leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

#   zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

#   last = tf.keras.layers.Conv2D(1, 4, strides=1,
#                                 kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

#   return tf.keras.Model(inputs=inp, outputs=last)

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
        # disc_loss2 = discriminator_loss(real_output, fake_output2)
        gen_loss = generator_loss(fake_generated_output,generated_images,real_images)

        # loss_euclidienne = tf.norm(tf.cast(generated_images,tf.float32) - tf.cast(real_images,tf.float32),ord="euclidean") # Calcul de la distance euclidienne
        # gen_loss = gen_loss + loss_euclidienne

        # if(disc_loss1 < disc_loss2): # Si le g??n??rateur rend un peu meilleur l'image
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
        print("Generator loss: {}".format(gen_loss))
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
    generator.summary()
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
    discriminator.summary()
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
    checkpoint_dir = "./Checkpoints/pix2pix bis/"
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


    train(liste_Tuple,batch_size, step, epochs=150)

