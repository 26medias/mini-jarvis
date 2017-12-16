import numpy as np
import cv2
import math

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

color_ok = (222, 169, 48)
color_not = (64, 58, 229)
pass_threshold = 0.2


# process a face and find the eyes, returning an array of cropped images of the eyes
def processEyes(img, eye1, eye2, threshold=0.4):
	ex1,ey1,ew1,eh1 = eye1
	ex2,ey2,ew2,eh2 = eye2
	area1 = ew1*eh1
	area2 = ew2*eh2
	diff = math.fabs(area1-area2)/min(area1,area2)
	if diff <= threshold:
		color = color_ok
	else:
		color = color_not
	cv2.rectangle(img,(ex1,ey1),(ex1+ew1,ey1+eh1),color,2)
	cv2.rectangle(img,(ex2,ey2),(ex2+ew2,ey2+eh2),color,2)
	return color, img[ey1:ey1+eh1, ex1:ex1+ew1], img[ey2:ey2+eh2, ex2:ex2+ew2]


# Process a face, finding the eyes and returning an array of cropped images (face, 2 eyes)
def processFace(img, gray, face, threshold=0.4):
	outputs = []
	x,y,w,h = face
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	color = color_not
	eyes = eye_cascade.detectMultiScale(roi_gray)
	
	outputs.append(img[y:y+h, x:x+w])
	
	if len(eyes) == 2:
		color, eye1Img, eye2Img = processEyes(roi_color, eyes[0], eyes[1], threshold)
		outputs.append(eye1Img)
		outputs.append(eye2Img)
	
	# Highlight the face no matter what
	cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
	return outputs



def stackAndResizeImages(images, size=250.0):
	output = []
	c = 0
	for image in images:
		# Resize the image
		resized = resizeToHeight(image, size)
		
		if c==0:
			output = resized
		else:
			output = np.hstack((output, resized))
		c = c+1
	return output



def resizeToHeight2(image, size=250):
	r = 250.0 / image.shape[1]
	dim = (250, int(image.shape[0] * r))
	#print dim
	#dim = (image.shape[1]*(size/image.shape[1]), image.shape[0]*(size/image.shape[1]))
	#resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def resizeToHeight(image, size=250):
	dim = (int(float(image.shape[1])*(float(size)/float(image.shape[0]))), int(size))
	print dim
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized
	
def resizeToWidth(image, size=250):
	dim = (int(size), int(float(image.shape[1])*(float(size)/float(image.shape[1]))))
	print dim
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized



while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	images = []
	
	resized = resizeToHeight(img, 250)
	images.append(resized)
	
	
	if len(faces)>0:
		for face in faces:
			faceParts = processFace(img, gray, face, pass_threshold)
			for facePart in faceParts:
				images.append(facePart)
	
	#displayImage = stackAndResizeImages(images, 250)
	
	if len(images)>0:
		displayImage = stackAndResizeImages(images, 250)
	else:
		displayImage = resizeToHeight(img, 250)
	#imgResized = resizeToHeight(img, 250)
	#cv2.imshow('displayImage',imgResized)
	#displayImage = stackAndResizeImages([img], 250)
	
	cv2.imshow('displayImage',displayImage)
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()
	
	