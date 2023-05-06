import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
imgsize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
folder = "DATA/C"
counter = 1
classifier = Classifier("keras_model.h5","labels.txt")
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","W","X","Y","Z"]
while True:
    succes,img = cap.read()
    imgOutput = img.copy()
    hands,img = detector.findHands(img)
    try:
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']
            imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
            imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255
            aspectRatio = h/w
            if aspectRatio>1:
                k = imgsize/h
                wcal = math.ceil(k*w)
                imgResize = cv2.resize(imgcrop,(wcal,imgsize))
                wGap = math.ceil((imgsize-wcal)/2)
                imgWhite[:,wGap:wGap+wcal] = imgResize
                prediction,index = classifier.getPrediction(imgWhite,draw=None)
            else:
                k = imgsize/w
                hcal = math.ceil(k*h)
                imgResize = cv2.resize(imgcrop,(imgsize,hcal))
                hGap = math.ceil((imgsize-hcal)/2)
                imgWhite[hGap:hGap+hcal,:] = imgResize
                prediction,index = classifier.getPrediction(imgWhite,draw=None)
               
            cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(239,125,51),4)
            cv2.rectangle(imgOutput,(x-2,y-60),(x+2+w,y),(239,125,51),cv2.FILLED)
            t = labels[index]
            t = t +" "+ str(prediction[index])[:4]
            cv2.putText(imgOutput,t,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(41,199,252),2)
            cv2.imshow("HAND",imgcrop)
            cv2.imshow("WHITE",imgWhite)
    except:
        pass
    
    cv2.imshow("WINDOW",imgOutput)
    k = cv2.waitKey(1)
    if k == 27:
        break