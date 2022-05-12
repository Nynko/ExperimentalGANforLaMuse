# Source : https://towardsdatascience.com/what-is-npy-files-and-why-you-should-use-them-603373c78883
# https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import os
import numpy as np
import tensorflow as tf
import argparse

height = 1024
width = 1024

## Load all the images in the folder Images and save them in a npy file
def LoadImagesToNpy(num):
    path_img= '../Images/'
    path_Collage = "../Images/Collage/"
    path_Real = '../Images/Real/'

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

    #  Load all the images in the folder Images/Collage
    images_collage = []
    for filename in os.listdir(path_Collage):
        if(num <= 0):
            break
        img = load_img(path_Collage+filename)
        #Resize the image
        img = img_to_array(img)
        img = tf.image.resize_with_pad(img,height,width)
        images_collage.append(img)
        num = num - 1
        

    images_collage = np.array(images_collage,dtype=object).astype('float32')

    # Load all the images in the folder Images/Real
    images_real = []
    for filename in os.listdir(path_Real):
        img = load_img(path_Real+filename)
        img = img_to_array(img)
        img = tf.image.resize_with_pad(img,height,width)
        images_real.append(img)
    
    images_real = np.array(images_real,dtype=object).astype('float32')
    
    print(type(images_collage))
    print(type(images_real))
    
    # Save the images in a npy file
    np.save(path_img+'Collage.npy', images_collage, True)
    np.save(path_img+'Real.npy', images_real, True)

    return None


if __name__ == "__main__":
    # Add number of images args
    parser = argparse.ArgumentParser(description='Create NPY files')
    parser.add_argument("-n","--number", help="Number of images to generate", required=True)
    args = parser.parse_args()
    num = args.number
    LoadImagesToNpy(int(num))