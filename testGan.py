from  BlendGan import *
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np
import tensorflow as tf
import os
import argparse



BATCH_SIZE = 100
IMG_SHAPE = (256, 256, 3)
EPOCHS= 100

height = IMG_SHAPE[0]
width = IMG_SHAPE[1]

def TestGan(img_path,output):
    # Load the images
    img = load_img(img_path)
    img = img_to_array(img)
    img = tf.image.resize_with_pad(img,height,width)/127.5 -1
    img = np.array(img,dtype=object).astype('float64')

    # Show image 
    img2 = (img + 1 ) * 127.5
    img2 = array_to_img(img2, scale=False)
    img2.show()


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

    if(latest != None):
        print("Checkpoint found")
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        print("Weights loaded")
    else:
        # generator.load_weights('{}/weight_{}.h5'.format("./Checkpoints",99))
        print("Checkpoint not found")

    
    generated_image = generator(img, training=False)[0]
    # Denormalize the image
    generated_image = (generated_image + 1 ) * 127.5
    img = array_to_img(generated_image, scale=False)
    img.show()

    # Save the image
    image_name = img_path.split('/')[-1]
    number_ckpt = latest.split('-')[-1].split('.')[0]
    img.save('{}/{}_ckpt{}.png'.format(output,image_name,number_ckpt))



if __name__ == "__main__":
    # Add number of images args
    parser = argparse.ArgumentParser(description='Test image on the GAN')
    parser.add_argument("-i","--image", help="Path for the image to test on the GAN", required=True)
    parser.add_argument("-o","--output", help="Path for the folder where the generated images will be saved",default="./Generated Images")
    args = parser.parse_args()
    img_path = args.image
    output = args.output
    TestGan(img_path,output)


