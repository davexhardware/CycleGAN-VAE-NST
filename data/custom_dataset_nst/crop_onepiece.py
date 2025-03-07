import cv2
import os
import matplotlib.pyplot as plt
import shutil
dataroot= './celeba/onepiece/'
images= os.listdir(dataroot)
face_classifier = cv2.CascadeClassifier(
    "./data/custom_dataset_nst/haarcascade_frontalface_default.xml"
    )
facesroot= './celeba/onepiece_faces_extr+crop+res+2/'
os.makedirs(facesroot, exist_ok=True)
for image in images:
    img = cv2.imread(dataroot+image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80,80)
    )
    if face is ():
        pass
    else:
        print(img.shape)
        if isinstance(face,list):
            face= max( face, key= lambda x: x[2]*x[3])
        for (x, y, w, h) in face:
            if img.shape[0]/w < 5 and img.shape[1]/h<5:
                shutil.copy(dataroot+image, facesroot+image)
            else:
                ymin= int(max(0, y-h/2))
                ymax= int(min(img.shape[0], y+1.5*h))
                xmin= int(max(0, x-w/2))
                xmax= int(min(img.shape[1], x+1.5*w))
                img = img[ymin:ymax, xmin:xmax,:]
                if img.size==0:
                    print('empty')
                cv2.imwrite(facesroot+image.split('.')[0]+'_crop.jpg', img)