from PIL import Image
import os
import random
import cv2
import argparse

## Il faut adapté le main pour utiliser ces fonctions de manière à créer plein d'images avec les contours et le mask:
# Il faut a chaque fois crée : un fichier de masque et une image avec le masque incrusté (pour être sûr que c'est ok) et l'image original avec juste l'image collée dessus (afin de peut être entrainer un gan d'inpainting en plus, mais pas sur pas ouf..jsp)



def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def create_mask(img_path):
    # img_path ="./image.png"
    print(img_path)

    if not path.isfile(img_path):
        print("fichier non trouvé")
        exit(0)
    # load image with alpha channel
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    dimensions = img.shape
    # height, width, number of channels in image
    height = dimensions[0]
    width = dimensions[1]
    # print('Image Dimension    : ',dimensions)
    # print('Image Height       : ',height)
    # print('Image Width        : ',width)


    #increase the size of the image with transparent padding 
    ratio_padding = 0.2
    top_bottom = int(height*ratio_padding)
    left_right = int(width*ratio_padding)
    img2 = cv2.copyMakeBorder(img, top_bottom,top_bottom,left_right,left_right, borderType=cv2.BORDER_CONSTANT, value=(255,255,255,0))
    # cv2.imwrite("A_border.png", img2)
    # dimensions = img2.shape
    # print('Image Dimension    : ',dimensions)

    # extract alpha channel
    alpha = img2[:, :, 3]

    # threshold alpha channel
    alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(image=alpha, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)


    #Resize contour
    scale = 1.15
    cnt_scaled = scale_contour(contours[0], scale=scale)
    contours_externe = [cnt_scaled]

    scale2 = 0.9
    cnt_scaled2 = scale_contour(contours[0], scale=scale2)
    contours_interne = [cnt_scaled2]


    # Put image in black
    img_copy = img2.copy()
    new_witdh = img_copy.shape[1]
    new_height = img_copy.shape[0]
    for col in range(new_witdh):
        for row in range(new_height):
            img_copy[row, col, :] = [0, 0, 0,0]


    # cv2.drawContours(img_copy, cnt_scaled, contourIdx=0, color=(0, 0, 255,1), thickness=1,lineType=cv2.LINE_AA)

    cv2.drawContours(img_copy, contours_externe, 0, (255, 255, 255,255), thickness=cv2.FILLED) # draw big contours
    cv2.drawContours(img_copy, contours_interne, 0, (0, 0, 0,0), thickness=cv2.FILLED) # draw alpha intern contour

    # save output
    cv2.imwrite(img_path[:-4] + "alpha" +".png", alpha)
    cv2.imwrite(img_path[:-4] + "scaled" +".png",img2)
    cv2.imwrite(img_path[:-4] + "contour" +".png",img_copy)

    return img2,img_copy

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