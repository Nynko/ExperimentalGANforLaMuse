from PIL import Image
import os
import random
import cv2
import argparse

def paste_image_from_file(background,img_path, output):

    if not os.path.isdir(background):
        print("background non trouvé")
        exit(1)

    if not os.path.isdir(img_path):
        print("image non trouvé")
        exit(1)

    if not os.path.isdir(output):
        print("output non trouvé")
        exit(1)

    # For all files in the directory images
    for file in os.listdir(img_path):
        if file.endswith(".png"):
            img_path2 = os.path.join(img_path, file)
            # Choose a random background image
            back = os.listdir(background)
            back2 = os.path.join(background, random.choice(back))
            paste_image(back2,img_path2,output)

def paste_image(back,img_path,output):

    background = Image.open(back)
    foreground = Image.open(img_path)

    # Generate random coordinates
    if(background.size[0] > foreground.size[0]):
        x = random.randint(0, background.size[0] - foreground.size[0])
    else:
        x = 0
    if(background.size[1] > foreground.size[1]):
        y = random.randint(0, background.size[1] - foreground.size[1])
    else:
        y = 0

    try:
        background.paste(foreground, (x,y), foreground)
    except:
        try:
            Image.alpha_composite(background, foreground)
        except:
            print("Error: ", img_path, " ", back, " background size: ", background.size, " foreground size: ", foreground.size)

    # save output
    name = img_path.split("/")[-1]
    nameFolder = img_path.split("/")[-2]
    path = output + "/" + nameFolder+ name +".png"
    background.save(path)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get contours from an image')
    parser.add_argument("-b","--backgrounds", help="Path to the folder of the images to use as background", required=True)
    parser.add_argument("-i","--images", help="Path to the folder of images in pngs")
    parser.add_argument("-o","--output", help="Path to the folder of the output images")
    args = parser.parse_args()
    background = args.backgrounds
    img_path = args.images
    output = args.output
    paste_image_from_file(background,img_path,output)