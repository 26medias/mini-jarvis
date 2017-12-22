import cv2
 
def camshift(img1, img2, bb):
        hsv = cv2.cvtColor(img1, cv.CV_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        x0, y0, w, h = bb
        x1 = x0 + w -1
        y1 = y0 + h -1
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist_flat = hist.reshape(-1)
        prob = cv2.calcBackProject([hsv,cv2.cvtColor(img2, cv.CV_BGR2HSV)], [0], hist_flat, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        new_ellipse, track_window = cv2.CamShift(prob, bb, term_crit)
        return track_window
 
def face_track():
    cap = cv2.VideoCapture(0)
    img = cap.read()
    bb = (125,125,200,100) # get bounding box from some method
    while True:
        try:
            img1 = cap.read()
            bb = camshift(img1, img, bb)
            img = img1
            #draw bounding box on img1
            imshow("CAMShift",img1)
        except KeyboardInterrupt:
            break

face_track()