import os
import cv2
from PIL import Image
import numpy as np

target_width = 32
target_height = 32
face_padding = 0.2
first = True

def scale_down(image: Image) -> Image:
    '''resizes the image to the global target_* values'''
    im = image

    # scale the smaller side to 'target' px
    width,height = im.size
    if width < height:
        width_new = target_width
        height_new = width_new * height / width
    else:
        height_new = target_height
        width_new = height_new * width / height

    im = im.resize((int(width_new), int(height_new)), Image.ANTIALIAS)
    return im

def detect_face(image: 'cv2.Mat') -> '(x,y,w,h)':
    '''Detects the face bounding box for an image.'''

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier('preprocessing/haarcascade_frontalface_default.xml')

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # print("Found {0} faces!".format(len(faces)))

    assert len(faces) == 1, "Image does not contain one distinct face!"

    # TODO remove loop (kept to reduce git changes)
    for (x, y, w, h) in faces:
        w_padding = int(w * face_padding)
        h_padding = int(h * face_padding)

        x -= w_padding
        w += 2 * w_padding

        y -= h_padding
        h += 2 * h_padding

        # squarify the rectangle
        if h > w:
            # x -= int((h-w)/2)
            w = h
        else:
            y -= int((w-h)/2)
            h = w

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # center point of the face-rectangle
        c = (x,y,w,h)
        return c

    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)

def squarify(image: Image, coi) -> Image:
    im = image
    a = coi[3]

    # left, top, right, bottom
    crop_box = (
        coi[0],
        coi[1],
        coi[0] + a,
        coi[1] + a
    )

    im = im.crop(crop_box)
    return im

def pil_to_opencv_image(pil_image: Image) -> 'cv2.Mat':
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

def crop_image_to_face(filename):
    try:
        # print("preprocessing "+filename)
        im = Image.open('images/'+filename)

        cv_im = pil_to_opencv_image(im)
        c = detect_face(cv_im)

        im = squarify(im, c)
        im = scale_down(im)

        cv_im = pil_to_opencv_image(im)
        detect_face(cv_im)

        im.save('images/'+filename)
            
    except AssertionError:
        os.remove('images/'+filename)
        # print("no distinct face found, deleting image")

def crop_to_face():
    '''Crops all images in folder ./images/ to their faces.'''
    if not os.path.exists('images'):
        print('Folder does not exist.')
        return
    
    for filename in os.listdir('images'):
       crop_image_to_face(filename)


if __name__ =="__main__":
    crop_to_face()