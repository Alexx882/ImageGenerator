import os
import cv2
from PIL import Image

target_width = 32
target_height = 32
face_padding = 0.2
first = True

def scaleDown(filename):
    '''
    resizes the image to the global target_* values 
    :param filename: name of the image relative to /images/
    '''

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
    im.save('images/'+filename)

def detect(filename):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier('preprocessing/haarcascade_frontalface_default.xml')

    # convert to grayscale
    image = cv2.imread('images/'+filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # print("Found {0} faces!".format(len(faces)))

    c = None

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

    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
    
    if len(faces) == 1:
        return c
    
    raise AssertionError("Image does not contain any faces!")

def squarify(filename, coi):
    im = Image.open('images/'+filename)

    a = coi[3]

    # left, top, right, bottom
    crop_box = (
        coi[0],
        coi[1],
        coi[0] + a,
        coi[1] + a
    )

    im = im.crop(crop_box)

    # save the final result
    im.save('images/'+filename)

if __name__ =="__main__":
    if not os.path.exists('images'):
        os.mkdir('images')
    
    for filename in os.listdir(r'images'):
        try:
            print("preprocessing "+filename)
            c = detect(filename)
            squarify(filename, c)
            scaleDown(filename)
            detect(filename)
            
        except AssertionError:
            os.remove('images/'+filename)
            print("no faces found, deleting image")

        # crop(filename)