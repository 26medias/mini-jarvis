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


def getBlobFrom(gray, x, y):
	xs		= []
	ys		= []
	mapped	= [[ 0 for _x in range(0,gray.shape[1])] for _y in range(0,gray.shape[0])]
	
	edge = [(x, y)]
	while edge:
		newedge = []
		for (x, y) in edge:
			for (_x, _y) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
				if _x > gray.shape[1] or _x <= 0 or _y > gray.shape[0] or _y <= 0 or mapped[_y][_x]==1:
					continue
				mapped[_y][_x]	= 1
				if gray[_y][_x] > 0:
					newedge.append((_x, _y))
					xs.append(_x)
					ys.append(_y)
					#image[_y, _x] = [255,0,0]
		edge = newedge
	
	bx	= min(xs)
	by	= min(ys)
	bx2	= max(xs)
	by2	= max(ys)
	bw	= max(xs)-bx
	bh	= max(ys)-by
	
	return (bx,by,bw,bh,bx2,by2)
	


def blobs(image, size):
	
	gray		= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray		= cv2.blur(gray,(size,size))
	gray		= cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
	gray		= cv2.dilate(gray, None, iterations=max([0,int(size/2)]))
	
	(bx,by,bw,bh,bx2,by2) = getBlobFrom(gray, 297, 165)
	
	cv2.rectangle(image,(bx,by),(bx2,by2),color_ok,1)
	
	cv2.imshow("gray", gray)
	cv2.imshow("image", image)
	




image		= cv2.imread('images/movement.png')

blobs(image, 10);


cv2.waitKey(0)
cv2.destroyAllWindows()