

import argparse
import cv2
import numpy as np
from os import path

def paste_image_from_file(background,img_path, mask_path):

    if not path.isfile(background):
        print("background non trouvé")
        exit(1)

    if not path.isfile(img_path):
        print("image non trouvé")
        exit(1)

    if not path.isfile(mask_path):
        print("masque non trouvé")
        exit(1)

    # load background, image et masque
    back = cv2.imread(background, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Put the mask on the image
    img2 = cv2.addWeighted(img, 1, mask, 1, 0)


    # Paste the image on the background
    x_offset=y_offset=50
    y1, y2 = y_offset, y_offset + img2.shape[0]
    x1, x2 = x_offset, x_offset + img2.shape[1]

    alpha_s = img2[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        back[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] +
                                alpha_l * back[y1:y2, x1:x2, c])

    #Create mask of the size of a black background the size of back
    mask_bg = np.zeros(back.shape, dtype=np.uint8)
    mask_bg[:,:,3] = 255
    alpha_s = mask[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        mask_bg[y1:y2, x1:x2, c] = (alpha_s * mask[:, :, c] +
                                alpha_l * mask_bg[y1:y2, x1:x2, c])

    # save output
    cv2.imwrite(img_path[:-4] + "background" +".png", back)
    cv2.imwrite(img_path[:-4] + "background" +"_mask" +".png", mask_bg)

    


def paste_image(back,img,mask,path):


    # Put the mask on the image
    img2 = cv2.addWeighted(img, 1, mask, 1, 0)


    # Paste the image on the background
    x_offset=y_offset=50
    y1, y2 = y_offset, y_offset + img2.shape[0]
    x1, x2 = x_offset, x_offset + img2.shape[1]

    alpha_s = img2[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        back[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] +
                                alpha_l * back[y1:y2, x1:x2, c])

    #Create mask of the size of a black background the size of back
    if(back.shape[2] == 4):
        mask_bg = np.zeros(back.shape, dtype=np.uint8)
        mask_bg[:,:,3] = 255
        alpha_s = mask[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            mask_bg[y1:y2, x1:x2, c] = (alpha_s * mask[:, :, c] +
                                    alpha_l * mask_bg[y1:y2, x1:x2, c])
    elif(back.shape[2] == 3):
        shape = back.shape
        shape = (shape[0], shape[1], 4)
        mask_bg = np.zeros(shape, dtype=np.uint8)
        mask_bg[:,:,3] = 255
        alpha_s = mask[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            mask_bg[y1:y2, x1:x2, c] = (alpha_s * mask[:, :, c] +
                                    alpha_l * mask_bg[y1:y2, x1:x2, c])


    # save output
    cv2.imwrite(path[:-4] + "background" +".png", back)
    cv2.imwrite(path[:-4] + "background" +"_mask" +".png", mask_bg)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get contours from an image')
    parser.add_argument("-b","--background", help="Path to the image to use as background", required=True)
    parser.add_argument("-i","--image", help="Path to the image")
    parser.add_argument("-m","--mask", help="Path to the mask")
    args = parser.parse_args()
    background = args.background
    img_path = args.image
    mask_path = args.mask
    paste_image_from_file(background,img_path,mask_path)