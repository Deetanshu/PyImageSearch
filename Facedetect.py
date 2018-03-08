# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:02:20 2018

@author: deept
"""
#import statements:

import numpy as np
import argparse
import cv2

#Constructing the argument parsing part:
ap=argparse.ArgumentParser()
inpimg=input("Please enter path to input image: ")
#ap.add_argument("-i", "--image", required=True, help="Path to input image")
#ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe")
#ap.add_argument("-m","--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Load the serialized model from disk.
print("<<LOADING MODEL>>");
net=cv2.dnn.readNetFromCaffe("CaffeStuff/deploy.prototxt.txt", "CaffeStuff/res10_300x300_ssd_iter_140000.caffemodel")

#Load input image and construct input blob and resize to 300x300
image=cv2.imread(inpimg)
(h,w)=image.shape[:2]
blob=cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

#Pass blob through network and obtain detections
print("<<Computing Object Detections>>")
net.setInput(blob)
detections = net.forward()

#looping over detections
for i in range(0, detections.shape[2]):
    #get probability of prediction
    confidence = detections[0, 0, i, 2]
    
    #filter out weaker detections
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        #Draw box on faces
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY+10
        cv2.rectangle(image, (startX,startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
#Show Output:
cv2.imshow("Output: ",image)
cv2.waitKey(0)