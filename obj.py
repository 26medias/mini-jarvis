import math
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import jarvisutils


cap = cv2.VideoCapture(0)

while 1:
	# load the input image, resize it, and convert it to grayscale
	ret, image	= cap.read()
	
	images = []
	images.append(image)
	
	rects = []
	dlib.find_candidate_object_locations(image, rects, min_size=500)
	
	
	
	# loop over the face detections
	for i, rect in enumerate(rects):
		x = rect.left()
		y = rect.top()
		w = rect.right()-rect.left()
		h = rect.bottom()-rect.top()
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
	
	output = jarvisutils.stack_line(images, 800)
	
	cv2.imshow("Output", output)
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()