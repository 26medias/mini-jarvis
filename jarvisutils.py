import math
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import scipy.misc

color_ok = (222, 169, 48)
color_not = (64, 58, 229)
color_found = (0, 255, 0)

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
	
	return img_crop(large, int(float(x)/rw), int(float(y)/rh), int(float(w)/rw), int(float(h)/rh)), (int(float(x)/rw), int(float(y)/rh), int(float(w)/rw), int(float(h)/rh))


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
		cropped, (_x,_y,_w,_h) = cropFromSmaller(imageSmall, image, x, y, w, h)
		faces.append([cropped, (_x, _y, _w, _h), (x, y, w, h)])
	
	return faces


def toXY(i, w):
	y	= (i/w)^0;
	x	= i-(y*w);
	return (x, y)

#
# Blob Detection
#


def getBlobFrom(gray, x, y, analyzed):
	xs		= []
	ys		= []
	mapped	= [[ 0 for _x in range(0,gray.shape[1])] for _y in range(0,gray.shape[0])]
	
	edge = [(x, y)]
	analyzed[y][x]	= 1
	
	while edge:
		newedge = []
		for (x, y) in edge:
			for (_x, _y) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
				if _x >= gray.shape[1] or _x <= 0 or _y >= gray.shape[0] or _y <= 0 or mapped[_y][_x]==1:
					continue
				mapped[_y][_x]	= 1
				if gray[_y][_x] > 0:
					newedge.append((_x, _y))
					xs.append(_x)
					ys.append(_y)
					#image[_y, _x]		= [255,0,0]
					analyzed[_y][_x]	= 1
		edge = newedge
	
	if len(xs)==0 or len(ys)==0:
		return False, (0,0,0,0,0,0)
	
	
	
	bx	= min(xs)
	by	= min(ys)
	bx2	= max(xs)
	by2	= max(ys)
	bw	= max(xs)-bx
	bh	= max(ys)-by
	
	return True, (bx,by,bw,bh,bx2,by2)




def getUnseenPixel(image, memory):
	for cy in range(0,image.shape[0]):
		for cx in range(0,image.shape[1]):
			if memory[cy][cx]==1:
				continue
			if (image[cy][cx]>0 and memory[cy][cx]==0):
				memory[cy][cx]=1
				return True, (cx, cy)
	return False, (0, 0)




def getBlobs3(image, size=10, gridSize=10):
	blobs		= []
	gray		= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray		= cv2.blur(gray,(size,size))
	gray		= cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
	
	
	sum = (float(np.sum(gray))/255.0)/(image.shape[0]*image.shape[1])*100.
	
	
	iterations	= 10 #int(math.ceil(max([1,(5.0*(100-min([100,sum*3])))/100.0])))
	
	print int(sum),'% ', iterations
	
	gray		= cv2.dilate(gray, None, iterations=iterations)
	
	grids		= []
	gx			= 0
	gy			= 0
	gw			= int(math.ceil(float(image.shape[1])/float(gridSize)))
	gh			= int(math.ceil(float(image.shape[0])/float(gridSize)))
	
	mini		= [[ 0 for _x in range(0,gw)] for _y in range(0,gh)]
	memory		= [[ 0 for _x in range(0,gw)] for _y in range(0,gh)]
	
	mini		= np.zeros([gh,gw,1])
	
	
	#print gw, gh
	
	for gy in range(0,gh):
		for gx in range(0,gw):
			#print gray.shape[0],gy+gridSize
			_y 		= gy*gridSize
			_x 		= gx*gridSize
			_w 		= min([gray.shape[1],gx*gridSize+gridSize])-_x
			_h 		= min([gray.shape[0],gy*gridSize+gridSize])-_y
			
			grid	= gray[_y:_y+_h, _x:_x+_w]
			sum = np.sum(grid)
			if sum>0:
				mini[gy][gx]	= 1
			else:
				memory[gy][gx]	= 1
	
	#print mini
	#print memory
	
	found = True
	while found:
		found, (x, y)			= getUnseenPixel(mini, memory)
		#print "Found: ", found, x, y, "mem:", memory[y][x]
		if found:
			hasBlobs, (bx,by,bw,bh,bx2,by2)	= getBlobFrom(mini, x, y, memory)
			if hasBlobs:
				blobs.append(((bx*gridSize,by*gridSize,bw*gridSize+gridSize,bh*gridSize+gridSize), (bx,by,bw,bh)))
	return blobs, mini
	#return blobs













def getBlobs2(image, gridSize=20, size=5):
	blobs		= []
	gray		= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray		= cv2.blur(gray,(size,size))
	gray		= cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
	gray		= cv2.dilate(gray, None, iterations=max([0,int(size/2)]))
	
	grids		= []
	gx			= 0
	gy			= 0
	gw			= int(math.ceil(float(image.shape[1])/float(gridSize)))
	gh			= int(math.ceil(float(image.shape[0])/float(gridSize)))
	
	mini		= [[ 0 for _x in range(0,gw)] for _y in range(0,gh)]
	memory		= [[ 0 for _x in range(0,gw)] for _y in range(0,gh)]
	
	mini		= np.zeros([gh,gw,1])
	
	
	#print gw, gh
	
	for gy in range(0,gh):
		for gx in range(0,gw):
			#print gray.shape[0],gy+gridSize
			_y 		= gy*gridSize
			_x 		= gx*gridSize
			_w 		= min([gray.shape[1],gx*gridSize+gridSize])-_x
			_h 		= min([gray.shape[0],gy*gridSize+gridSize])-_y
			
			grid	= gray[_y:_y+_h, _x:_x+_w]
			sum = np.sum(grid)
			if sum>0:
				mini[gy][gx]	= 1
			else:
				memory[gy][gx]	= 1
	
	#print mini
	#print memory
	
	found = True
	while found:
		found, (x, y)			= getUnseenPixel(mini, memory)
		#print "Found: ", found, x, y, "mem:", memory[y][x]
		if found:
			hasBlobs, (bx,by,bw,bh,bx2,by2)	= getBlobFrom(mini, x, y, memory)
			if hasBlobs:
				blobs.append(((bx*gridSize,by*gridSize,bw*gridSize+gridSize,bh*gridSize+gridSize), (bx,by,bw,bh)))
	return blobs, mini
	#return blobs














def getBlobs(image, gridSize=20, size=5):
	blobs		= []
	gray		= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray		= cv2.blur(gray,(size,size))
	gray		= cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
	gray		= cv2.dilate(gray, None, iterations=max([0,int(size/2)]))
	
	analyzed	= [[ 0 for _x in range(0,gray.shape[1])] for _y in range(0,gray.shape[0])]
	npAnalyzed	= np.array(analyzed)
	
	grids		= []
	gx			= 0
	gy			= 0
	gw			= int(math.ceil(float(image.shape[1])/float(gridSize)))
	gh			= int(math.ceil(float(image.shape[0])/float(gridSize)))
	
	print gw, gh
	
	for gy in range(0,gh):
		for gx in range(0,gw):
			#print gray.shape[0],gy+gridSize
			_y 		= gy*gridSize
			_x 		= gx*gridSize
			_w 		= min([gray.shape[1],gx*gridSize+gridSize])-_x
			_h 		= min([gray.shape[0],gy*gridSize+gridSize])-_y
			grid	= gray[_y:_y+_h, _x:_x+_w]
			
			sum = np.sum(grid)
			
			if sum==0:
				npAnalyzed[_y:_y+_h, _x:_x+_w] = 1
				cv2.rectangle(image,(_x,_y),(_x+_w,_y+_h),color_not,1)
			else:
				cv2.rectangle(image,(_x,_y),(_x+_w,_y+_h),color_ok,1)
			
			grids.append(grid)
	
	analyzed = npAnalyzed.tolist()
	
	#sumA = np.sum(analyzed)
	#print "sumA: ", sumA
	
	#return []
	
	found = True
	while found:
		found, (x, y)			= getUnseenPixel(gray, analyzed)
		print found, x, y
		if found:
			hasBlobs, (bx,by,bw,bh,bx2,by2)	= getBlobFrom(gray, x, y, analyzed)
			blobs.append((bx,by,bw,bh))
			#cv2.rectangle(image,(bx,by),(bx2,by2),color_found,1)
	return blobs
	#cv2.imshow("image", image)
	#cv2.imshow("gray", 	gray)
