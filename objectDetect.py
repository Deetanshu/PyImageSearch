# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:08:27 2018

@author: deept
"""

#imports
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
 
#Argument Parser made anaconda-friendly:
prototxt="CaffeStuff/MobileNetSSD_deploy.prototxt.txt"
cmodel="CaffeStuff/MobileNetSSD_deploy.caffemodel"
ap=argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum probability")
args=vars(ap.parse_args())

#Create list of classes and color set:
classes=["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors=np.random.uniform(0,255, size=(len(classes), 3))

#Load pre-trained model:
print("Loading Model: ")
net = cv2.dnn.readNetFromCaffe(prototxt, cmodel)

print("Starting video stream:")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps=FPS().start()

#Loop for frames:
while True:
    frame=vs.read()
    frame=imutils.resize(frame, width=400)
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections=net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence > args["confidence"]:
            idx = int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY)=box.astype("int")
            
            label="{}: {:.2f}%".format(classes[idx], confidence*100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
            y=startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    #show images, within the bigger loop:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)&0xFF
    
    if key == ord("q"):
        break
    fps.update()

#cleanup
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()