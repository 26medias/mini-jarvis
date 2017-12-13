import numpy as np
import cv2
import math

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

color_ok = (222, 169, 48)
color_not = (64, 58, 229)

while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	if len(faces)==0:
		# No face detected
		r = 1024.0 / img.shape[1]
		dim = (1024, int(img.shape[0] * r))
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		cv2.imshow('resized',resized)
	else:
		x,y,w,h = faces[0]
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		color = color_not
		eyes = eye_cascade.detectMultiScale(roi_gray)
		if len(eyes) == 2:
			ex1,ey1,ew1,eh1 = eyes[0]
			ex2,ey2,ew2,eh2 = eyes[1]
			area1 = ew1*eh1
			area2 = ew2*eh2
			diff = math.fabs(area1-area2)/min(area1,area2)
			if diff <= 0.3:
				color = color_ok
			cv2.rectangle(roi_color,(ex1,ey1),(ex1+ew1,ey1+eh1),color,2)
			cv2.rectangle(roi_color,(ex2,ey2),(ex2+ew2,ey2+eh2),color,2)
			
		cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
		
		r = 1024.0 / img.shape[1]
		dim = (1024, int(img.shape[0] * r))
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		cv2.imshow('resized',resized)
	
	
		#cropped = img[x:y, x+w:y+h]
		#r = 1024.0 / w
		#dim = (1024, int(w * r))
		#resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
		#cv2.imshow('resized',resized)
		#cv2.imshow('resized',cropped)
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()
	