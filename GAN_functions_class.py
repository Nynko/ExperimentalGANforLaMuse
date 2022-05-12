
import os
import time
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img



## Variables globales
channels = 3
img_shape = (256,256,channels)

path_img= 'Images/'
path_Collage = "Images/Collage/"
path_Real = 'Images/Real/'

LAMBDA = 1


class GanFunctionsClass:
  def __init__(self, batchsize, img_shape):
      self.batchsize = batchsize
      self.img_shape = img_shape
      self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  # =================================================================================== #
  #               1. Define the discriminator and generator losses                      #
  # =================================================================================== # 

  def discriminator_loss(self,disc_real_output, disc_generated_output):
    real_loss = self.cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = self.cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


  def generator_loss(self,output_discriminateur): 
      loss = self.cross_entropy(tf.ones_like(output_discriminateur), output_discriminateur)
      return loss



  # =================================================================================== #
  #               2. Define the  generator                                              #
  # =================================================================================== #  

  def downsample(self,filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

  def upsample(self,filters, size, apply_dropout=False):
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


  def build_generator(self,img_shape):
    inputs = tf.keras.layers.Input(shape=img_shape)

    down_stack = [
      self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
      self.downsample(128, 4),  # (batch_size, 64, 64, 128)
      self.downsample(256, 4),  # (batch_size, 32, 32, 256)
      self.downsample(512, 4),  # (batch_size, 16, 16, 512)
      self.downsample(512, 4),  # (batch_size, 8, 8, 512)
      self.downsample(512, 4),  # (batch_size, 4, 4, 512)
      self.downsample(512, 4),  # (batch_size, 2, 2, 512)
      self.downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
      self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
      self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
      self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
      self.upsample(512, 4),  # (batch_size, 16, 16, 1024)
      self.upsample(256, 4),  # (batch_size, 32, 32, 512)
      self.upsample(128, 4),  # (batch_size, 64, 64, 256)
      self.upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(channels, 4,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


  # =================================================================================== #
  #               3. Define the discriminator                                           #
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


  # =================================================================================== #
  #               4. UTILS - Récupération des batchs, loads des images...               #
  # =================================================================================== # 

  def getTupleRandom(self,batchsize):
      listCollages = os.listdir("./Images/Collage")
      listReal = os.listdir("./Images/Real")
      random.shuffle(listCollages)
      random.shuffle(listReal)

      minGlobal = min(len(listCollages), len(listReal))
      compt_batch = 0
      listTuple = []
    
      while compt_batch*batchsize < minGlobal :

          images_real = listReal[compt_batch*batchsize:(compt_batch+1)*batchsize]
          images_collage = listCollages[compt_batch*batchsize:(compt_batch+1)*batchsize]
          
          tuple = (
            images_real,
            images_collage
          )
          listTuple.append(tuple)
          compt_batch += 1

      return listTuple


  def LoadImages(self,batch,batchsize):

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
      

  def showLosses(self,liste_losses):
      plt.figure(figsize=(10,10))
      loss_gen = []
      loss_disc = []
      for i in range(len(liste_losses)):
          loss_gen.append(liste_losses[i][0])
          loss_disc.append(liste_losses[i][1])
      plt.plot(loss_gen,label="Generator")
      plt.plot(loss_disc,label="Discriminator")
      plt.legend(loc="upper left")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.title("Losses")
      plt.show()
      
      
      
  # =================================================================================== #
  #               5. Fonctions d'entrainements                                          #
  # =================================================================================== # 



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
          gen_loss = self.generator_loss(fake_output)


      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

      return gen_loss, disc_loss

  def train(self,checkpoint, checkpoint_prefix,checkpoint_dir,generator, discriminator,generator_optimizer, discriminator_optimizer,listeTuple,batch_size,step, loss_liste ,epochs):
      if epochs > len(listeTuple):
          print("Epochs too high")
          exit(1)

      step2 = step.numpy().item()
      epoch = step2 

      while epoch <= epochs:
          start = time.time()

          batch = listeTuple[epoch]
          batch_real, batch_fake = self.LoadImages(batch,batch_size)
          gen_loss, disc_loss = self.train_step(generator, discriminator,generator_optimizer, discriminator_optimizer,batch_real,batch_fake)

          # Print model losses
          print("")
          print("Generator loss: {}".format(gen_loss))
          print("Discriminator loss: {}".format(disc_loss))

          # Adding the losses to the list
          loss_liste.append((gen_loss,disc_loss))


          # Save the model every 10 epochs
          step.assign(epoch)
          if epoch % 10 == 0:
              checkpoint.save(file_prefix = checkpoint_prefix)
              print("Saved checkpoint")
              #save the losses
              with open(checkpoint_dir+"loss_liste.npy", 'wb+') as f:
                np.save(f,loss_liste)
              print("Saved losses")
          print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
          epoch += 1

      self.showLosses(loss_liste)
    
      return None 
