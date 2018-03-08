# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:43:09 2018

@author: deept
"""
#imports
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

#Skipping argparses and making the paths iron:
prototxt="CaffeStuff/deploy.prototxt.txt"
cmodel="CaffeStuff/res10_300x300_ssd_iter_140000.caffemodel"

#Add confidence in case:
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum prediction")
args= vars(ap.parse_args())

#Load Model:
print("Loading the Model")
net = cv2.dnn.readNetFromCaffe(prototxt, cmodel)

#Start Video Stream:
print("Starting Video Stream: ")
vs= VideoStream(src=0).start()
time.sleep(2.0)

#Frame loop:
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    #loop over detections now:
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence < args["confidence"]:
            continue
        
        #The box making part:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
    
    #Show the output (and since it's a frame keep it in the loop)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key==ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()