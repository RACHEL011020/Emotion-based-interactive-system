#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:33:01 2022

@author: mgongo00
 from: https://stackoverflow.com/questions/64126876/how-to-detect-multiple-faces-from-the-same-image

Get OpenCV by:
    > pip install opencv-python
    
Had to find the file "haarcascade_frontalface_default.xml" in the system and copy it to
the desired folder ("Modles" in this case)

"""

import cv2
import numpy as np

# Load some pre-trained data on face frontal from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('data/haarcascade_files/haarcascade_frontalface_default.xml')
# trained_face_data = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('Data/pipol.png')
img = cv2.imread('IMG_4497.jpg')

# Must convert to greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gey Img', grayscaled_img)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

img_crop = []

print('Faces seen.')

counter = 0
# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_crop.append(img[y:y + h, x:x + w])
    
print('Rectangles done')
#exit()

for counter, cropped in enumerate(img_crop):
    cv2.imshow('Cropped', cropped)
    cv2.imwrite("Data/aud_result_{}.png".format(counter), cropped)

"""    
# This locks the system until a keystroke, to view  when testing.
# does not work in spyder, just when run from a terminal.
    cv2.imshow('Cropped', img_crop[counter])
    cv2.waitKey(0)
"""    
print('Faces saved.')
#exit()


# -----------------------------------------------------------------------------


# End.
