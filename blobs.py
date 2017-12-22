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




image	= cv2.imread('images/movement.png')

blobs	= jarvisutils.getBlobs2(image, 5);
print blobs

for blob in blobs:
	(x,y,w,h) = blob
	#print x,y,w,h
	cv2.rectangle(image,(x,y),(x+w,y+h),color_found,1)
	#images.append(jarvisutils.img_crop(image, x, y, w, h))

cv2.imshow("image", image)

#images	= []
#images.append(image)
#for blob in blobs:
#	(x,y,w,h) = blob
#	print x,y,w,h
#	images.append(jarvisutils.img_crop(image, x, y, w, h))
#	cv2.rectangle(image,(x,y),(x+w,y+h),color_found,1)


#output = jarvisutils.stack_line(images, 200)
#cv2.imshow("Output", output)

cv2.waitKey(0)
cv2.destroyAllWindows()