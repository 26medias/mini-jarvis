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

cap = cv2.VideoCapture(0)

faceFound = []
f = 0

foundBefore = 0

def trackThis(image, coords):
	track_window = coords
	x,y,w,h		= track_window
	# set up the ROI for tracking
	hsv_roi		= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	roi			= image[y:y+h, x:x+w]
	mask		= cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
	roi_hist	= cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	
	return roi_hist, track_window, roi


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit	= ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

trackingImage = []
roi_hist = []
track_window = (0,0,0,0)

while 1:
	# load the input image, resize it, and convert it to grayscale
	ret, image	= cap.read()
	image		= imutils.resize(image, width=800)
	imageClean	= imutils.resize(image, width=800)
	
	faces	= jarvisutils.getFaces(imageClean, 300)
	
	images	= [image]
	
	cv2.putText(image,"Frame #%d, %d faces" % (f, len(faces)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
	
	if f>20 and len(faces)>0 and len(faceFound)==0 and len(trackingImage)==0:
		cropped, (x,y,w,h), (sx,sy,sw,sh) = faces[0]
		faceFound = [cropped, (x,y,w,h)]
		roi_hist, track_window, roi	= trackThis(imageClean, (x,y,w,h))
		trackingImage = roi
		#cv2.putText(roi,"-TARGET-", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
		print "Tracking:", (x,y,w,h)
	
	if len(trackingImage)>0:
		images.append(trackingImage)
	
	# loop over the face detections
	for face in faces:
		cropped, (x,y,w,h), (sx,sy,sw,sh) = face
		cv2.putText(cropped,"Face", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
		images.append(cropped)
		cv2.rectangle(image,(x,y),(x+w,y+h),color_not,1)
		
		
	if len(faceFound)>0 and f%10==0:
		#images.append(faceFound[0])
		
		hsv	= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		dst	= cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		
		
		# apply meanshift to get the new location
		ret, track_window = cv2.CamShift(dst, track_window, term_crit)
		
		# Draw it on image
		#pts		= cv2.boxPoints(ret)
		#pts		= np.int0(pts)
		#img2	= cv2.polylines(image,[pts],True, 255,2)
		#x,y,w,h = track_window
		#img2 = cv2.rectangle(image, (x,y), (x+w,y+h), 255,2)
	
		
	print "track_window: ", track_window
	(tx,ty,tw,th) = track_window
	#print "roi_hist: ", roi_hist
	cv2.rectangle(image,(tx,ty),(tx+tw,ty+th),color_ok,1)
	
	
	output = jarvisutils.stack_line(images, 400)
	cv2.imshow("Output", output)
	
	f = f+1
	foundBefore = len(faces)
	
	
	
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
		cap.release()
		cv2.destroyAllWindows()