import math
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import jarvisutils


cap = cv2.VideoCapture(0)

color_ok = (222, 169, 48)
color_not = (64, 58, 229)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("haar/shape_predictor_68_face_landmarks.dat")



def processFace(gray, image, rect):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	
	pitch, isLooking = facePitch(image, shape)
	#print (pitch, isLooking)
	
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	
	# Crop the face
	_x, _y, _w, _h = jarvisutils.pointsbb(shape)
	croppedFace = jarvisutils.img_crop(image, _x, _y, _w, _h)
	
	if isLooking:
		cv2.rectangle(image,(_x,_y),(_x+_w,_y+_h),color_ok,2)
	else:
		cv2.rectangle(image,(_x,_y),(_x+_w,_y+_h),color_not,2)
	
	return croppedFace, (_x, _y, _w, _h)


def facePitch(image, points):
	lineX = points[27][0]-points[30][0]
	lineY = points[27][1]-points[30][1]
	
	# Find the top lip box
	lip_x, lip_y, lip_w, lip_h = jarvisutils.pointsbb([points[32], points[33], points[34]])
	cv2.rectangle(image,(lip_x,lip_y),(lip_x+lip_w,lip_y+lip_h),color_ok,2)
	# Center of the lip box
	mid_x		= int(float(lip_x)+float(lip_w)/2)
	mid_y		= int(float(lip_y)+float(lip_h)/2)
	intercept_x	= points[30][0]+int(float(mid_y-points[30][1])*float(lineX)/float(lineY))
	
	#print (lineX, lineY, float(lineX)/float(lineY), intercept_x)
	cv2.circle(image, (intercept_x, mid_y), 1, (255, 255, 255), -1)
	
	return mid_x-intercept_x, points[32][0] <= intercept_x and intercept_x <= points[34][0]
	

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


while 1:
	# load the input image, resize it, and convert it to grayscale
	ret, image	= cap.read()
	imageSmall	= imutils.resize(image, width=400)
	imageHD		= image
	gray		= cv2.cvtColor(imageSmall, cv2.COLOR_BGR2GRAY)
	gray		= cv2.equalizeHist(gray)
	gray2		= cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
	#equ			= cv2.equalizeHist(gray)
	#gray3		= cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
	#gray3		= adjust_gamma(gray2, 1)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	
	images = []
	images.append(imageSmall)
	images.append(gray2)
	#images.append(gray2)
	#images.append(gray3)
	#images.append(equ)
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		faceCrop, (x,y,w,h) = processFace(gray, imageSmall, rect)
		largeCrop	= jarvisutils.cropFromSmaller(imageSmall, imageHD, x,y,w,h)
		#images.append(faceCrop)
		images.append(largeCrop)
	
	output = jarvisutils.stack_line(images, 400)
	
	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", output)
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()