from PasteImage import paste_image
from Contour    import create_mask
import cv2
import argparse
from os import path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get contours from an image')
    parser.add_argument("-i","--image", help="Path to the image")
    parser.add_argument("-b","--background", help="Path to the image to use as background", required=True)
    args = parser.parse_args()
    background = args.background
    img_path = args.image
    (img,mask) = create_mask(img_path)

    if not path.isfile(background):
        print("background non trouvé")
        exit(1)
    else:
        path = img_path
        back = cv2.imread(background, cv2.IMREAD_UNCHANGED)
        paste_image(back,img,mask,path)


