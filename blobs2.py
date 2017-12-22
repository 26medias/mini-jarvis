import math
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import jarvisutils


color_ok = (222, 169, 48)
color_not = (64, 58, 229)
color_found = (0, 255, 0)

cap = cv2.VideoCapture(0)

previous = []
started = False
f = 0


while 1:
	#print "--------------------"
	# load the input image, resize it, and convert it to grayscale
	ret, image	= cap.read()
	frame		= imutils.resize(image, width=500)
	gray		= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray		= cv2.GaussianBlur(gray, (5, 5), 0)
	
	if not started:
		previous	= gray
		started		= True
	
	
	frameDelta	= cv2.absdiff(previous, gray)
	thresh		= cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
	thresh		= cv2.dilate(thresh, None, iterations=1)
	thresh		= cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
	
	# Find the blobs
	#blobs	= jarvisutils.getBlobs2(thresh, 50, 20);
	
	
	
	blobs, mini	= jarvisutils.getBlobs3(thresh, 10);
	#print "Blobs: ", blobs
	
	for blob in blobs:
		(x,y,w,h), (x2,y2,w2,h2) = blob
		#print x,y,w,h
		cv2.rectangle(frame,(x,y),(x+w,y+h),color_not,3)
		
		#images.append(jarvisutils.img_crop(image, x, y, w, h))
	
	
	
	
	#for blob in blobs:
		#(x,y,w,h) = blob
		#print x,y,w,h
		#cv2.rectangle(image,(x,y),(x+w,y+h),color_found,1)
		#images.append(jarvisutils.img_crop(image, x, y, w, h))
	
	cv2.imshow("frame", frame)
	cv2.imshow("mini", mini)
	#cv2.imshow("thresh", thresh)
	
	
	f = f+1
	
	if f%10:
		previous	= gray
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()