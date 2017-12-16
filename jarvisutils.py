import cv2
import numpy as np
import dlib
import imutils

detector = dlib.get_frontal_face_detector()

def resizeToHeight(image, size=250):
	dim = (int(float(image.shape[1])*(float(size)/float(image.shape[0]))), int(size))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized


def resizeToWidth(image, size=250):
	dim = (int(size), int(float(image.shape[1])*(float(size)/float(image.shape[1]))))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized


def stack_line(images, size=250.0):
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


def stack_col(images, size=250.0):
	output = []
	c = 0
	for image in images:
		# Resize the image
		resized = resizeToWidth(image, size)
		
		if c==0:
			output = resized
		else:
			output = np.vstack((output, resized))
		c = c+1
	return output


def img_crop(image, x, y, w, h):
	return image[y:y+h, x:x+w]


def pointsbb(points):
	xs = []
	ys = []
	for (x,y) in points:
		xs.append(x)
		ys.append(y)
	_x = min(xs)
	_y = min(ys)
	_w = max(xs)-_x
	_h = max(ys)-_y
	#print (_x, _y, _w, _h)
	#
	return _x, _y, _w, _h


def cropFromSmaller(small, large, x, y, w, h):
	rw = float(small.shape[0])/float(large.shape[0])
	rh = float(small.shape[1])/float(large.shape[1])
	
	#print (small.shape, large.shape, rw, rh, int(float(x)/rw), int(float(y)/rh), int(float(w)/rw), int(float(h)/rh))
	
	return img_crop(large, int(float(x)/rw), int(float(y)/rh), int(float(w)/rw), int(float(h)/rh))


def getFaces(image, quality=250):
	imageSmall	= imutils.resize(image, width=quality)
	gray		= cv2.cvtColor(imageSmall, cv2.COLOR_BGR2GRAY)
	gray		= cv2.equalizeHist(gray)
	rects		= detector(gray, 1)
	
	faces = []
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		x = rect.left()
		y = rect.top()
		w = rect.right()-rect.left()
		h = rect.bottom()-rect.top()
		cropped = cropFromSmaller(imageSmall, image, x, y, w, h)
		faces.append([cropped, (x, y, w, h)])
	
	return faces