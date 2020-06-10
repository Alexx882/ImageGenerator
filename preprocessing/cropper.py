import os
from PIL import Image

target_width = 32
target_height = 32
first = True

def crop(filename):
    '''
    Resizes and crops the image
    :param filename: name of the image relative to /images/
    '''

    print("cropping: "+filename)

    im = Image.open('images/'+filename)

    # scale the smaller side to 'target' px
    width,height = im.size
    if width < height:
        width_new = target_width
        height_new = width_new * height / width
    else:
        height_new = target_height
        width_new = height_new * width / height

    im = im.resize((int(width_new), int(height_new)), Image.ANTIALIAS)

    # crop image to target_height x target_width
    width,height = im.size
    if width < height:
        diff = height - width

        # top, left, right, bottom
        crop_box = (0, target_width, diff / 2, diff / 2 + target_height)
    else:
        diff = width - height

        crop_box = (diff / 2, diff / 2 + target_width, 0, target_height)
    
    im = im.crop(crop_box)

    # save the final result
    im.save('images/'+filename)


if __name__ =="__main__":
    if not os.path.exists('images'):
        os.mkdir('images')
    
    for filename in os.listdir(r'images'):
        crop(filename)