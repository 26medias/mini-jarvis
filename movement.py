import math
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import jarvisutils


cap = cv2.VideoCapture(0)

previous = []
started = False
f = 0

while 1:
	# load the input image, resize it, and convert it to grayscale
	ret, image	= cap.read()
	frame		= imutils.resize(image, width=500)
	gray		= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray		= cv2.GaussianBlur(gray, (5, 5), 0)
	
	if not started:
		previous	= gray
		started		= True
	
	cv2.imshow("image", image)
	
	frameDelta	= cv2.absdiff(previous, gray)
	thresh		= cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]
	thresh		= cv2.dilate(thresh, None, iterations=1)
	#(cnts, _)	= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	cv2.imshow("Output", thresh)
	
	f = f+1
	
	if f%10:
		previous	= gray
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()