import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
imgsize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
folder = "DATA/V"
counter = 1
while True:
    succes,img = cap.read()
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
                # imgcropshape = imgResize.shape
                wGap = math.ceil((imgsize-wcal)/2)
                imgWhite[:,wGap:wGap+wcal] = imgResize
            else:
                k = imgsize/w
                hcal = math.ceil(k*h)
                imgResize = cv2.resize(imgcrop,(imgsize,hcal))
                # imgcropshape = imgResize.shape
                hGap = math.ceil((imgsize-hcal)/2)
                imgWhite[hGap:hGap+hcal,:] = imgResize                

            cv2.imshow("HAND",imgcrop)
            cv2.imshow("WHITE",imgWhite)
    except:
        pass
    
    cv2.imshow("WINDOW",img)
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
        counter +=1
    elif k == 27:
        break