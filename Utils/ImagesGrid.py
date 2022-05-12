import math
from PIL import Image


def createImageAsAGrid(files,output):

    lenght = len(files)

    # Assuming 256*256 images
    height = 256
    width = 256

    ratio = 4
    height_new = height * ratio
    width_new = width * math.ceil(lenght /ratio)

    print(height_new,width_new)

    # Create a new image
    new_im = Image.new('RGB', (height_new, width_new))

    index = 0
    for i in range(0,height_new,height):
        for j in range(0,width_new,width):
            if(index >= lenght):
                print("index: ", index, " lenght: ", lenght)
                print("i: ", i, " j: ", j)
                break
            im = Image.open(files[index])
            im.thumbnail((height, width))
            new_im.paste(im, (i,j))
            index += 1
    name = files[0].split("/")[-1].split(".")[0]
    new_im.save(output + "/" + name + 'grid.png')

    return None