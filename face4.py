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
	
	faces	= jarvisutils.getFaces(image, 100)
	
	images	= [image]
	
	# loop over the face detections
	for face in faces:
		cropped, (x,y,w,h) = face
		images.append(cropped)
	
	output = jarvisutils.stack_line(images, 400)
	
	cv2.imshow("Output", output)
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()