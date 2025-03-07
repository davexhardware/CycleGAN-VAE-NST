import shutil
import os
import face_recognition
dataroot= './celeba/onepiece/'
images= os.listdir(dataroot)
dest_dir='./celeba/onepiece_face_recon/'
os.makedirs(dest_dir, exist_ok=True)
for image in images:
    if 'modified' not in image:
        img=face_recognition.load_image_file(dataroot+image)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations)==0:
            print(f'No face detected in {image}')
        else:
            print(f'Face detected in {image}')
            for i,face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = img[top:bottom, left:right]
                shutil.copy(dataroot+image, dest_dir+image.split('.')[0]+'_'+str(i)+'.jpg')
