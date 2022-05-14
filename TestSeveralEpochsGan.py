from  BlendGan import *
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np
import tensorflow as tf
import os
import argparse

from Utils.ImagesGrid import *



BATCH_SIZE = 100
IMG_SHAPE = (256, 256, 3)
EPOCHS= 100

height = IMG_SHAPE[0]
width = IMG_SHAPE[1]

"""
Create and save all the output images for one input image
Also create and save a grid of all the images.
"""
def TestGan(img_path,output):
    # Load the images
    img = load_img(img_path)
    img = img_to_array(img)
    img = tf.image.resize_with_pad(img,height,width)/127.5 -1
    img = np.array(img,dtype=object).astype('float64')

    # Show image 
    img2 = (img + 1 ) * 127.5
    img2 = array_to_img(img2, scale=False)
    # img2.show()
    img2.save(output+'.png')


    img = img.reshape(1,height,width,3)

    # Create a GanFunctionsClass object
    gan = GanFunctionsClass(BATCH_SIZE,IMG_SHAPE)


    # Load the model
    generator = gan.build_generator((height,width,3))
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator.compile(loss=gan.generator_loss, optimizer=generator_optimizer)
    discriminator = gan.build_discriminator(IMG_SHAPE)
    discriminator.compile(loss=[gan.discriminator_loss],
                                optimizer=discriminator_optimizer,
                                metrics=['accuracy'])

    ## Test d'une image sur le generateur
    #Si des fichiers de checkpoints existent, on les charge
    checkpoint_dir = './Checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    if(latest == None):
        print("Checkpoint not found")
        exit(1)

    
    # Iterating over each checkpoints
    for i in range(1,int(latest.split('-')[-1])):
        print("Checkpoint: ", i)
        checkpoint.restore(checkpoint_dir+"ckpt-{}".format(i)).expect_partial()

        generated_image = generator(img, training=False)[0]
        # Denormalize the image
        generated_image = (generated_image + 1 ) * 127.5
        img2 = array_to_img(generated_image, scale=False)
        # img.show()

        # Save the image
        image_name = img_path.split('/')[-1]
        img2.save('{}/{}_ckpt{}.png'.format(output,image_name,i))

    # Save the image as a grid 
    image_name = img_path.split('/')[-1]
    createImageAsAGrid(['{}/{}_ckpt{}.png'.format(output,image_name,i) for i in range(1,int(latest.split('-')[-1]))],output)

    return None



if __name__ == "__main__":
    # Add number of images args
    parser = argparse.ArgumentParser(description='Test image on the GAN')
    parser.add_argument("-i","--image", help="Path for the image to test on the GAN", required=True)
    parser.add_argument("-o","--output", help="Path for the folder where the generated images will be saved",required=True)
    args = parser.parse_args()
    img_path = args.image
    output = args.output
    TestGan(img_path,output)


