
# Sources:
# https://stackoverflow.com/questions/61918194/how-to-make-a-binary-mask-out-of-an-image-with-a-transparent-background
# https://stackoverflow.com/questions/66793516/how-increase-an-image-to-a-specific-size-by-adding-white-transparent-pixels-in-p
# https://www.tutorialkart.com/opencv/python/opencv-python-get-image-size/#:~:text=When%20working%20with%20OpenCV%20Python,of%20channels%20for%20each%20pixel.
# https://note.nkmk.me/en/python-opencv-numpy-alpha-blend-mask/
# https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
# 
# Fill contours :https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python

import argparse
import cv2
import numpy as np
from os import path


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get contours from an image')
    parser.add_argument("-i","--image", help="Path to the image")
    args = parser.parse_args()
    img_path = args.image
    create_mask(img_path)