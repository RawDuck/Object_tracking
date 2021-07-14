import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# variables
ix, iy, jx, jy = -1, -1, -1, -1
drawing = False
vid_file = "./sek2.mp4"
cap = cv.VideoCapture(vid_file)
_, img = cap.read()

def meanshift_own(I, startY, startX):
    M00, M01, M10 = 0, 0, 0
    print(startX, I.shape[0])
    print(startY, I.shape[1])
    for aa in range(startX, startX + 200):
        for bb in range(startY, startY + 200):
            M00 = M00 + I[aa, bb]
            M10 = M10 + I[aa, bb] * bb
            M01 = M01 + I[aa, bb] * aa
    xc = M01 / M00
    if xc + 100 > I.shape[0]:
        xc = I.shape[0] - 100
    yc = M10 / M00
    if yc + 100 > I.shape[1]:
        yc = I.shape[1] - 100
    return math.floor(yc) - 100, math.floor(xc) - 100


def draw_reactangle_with_drag(event, x, y, flags, param):
    global ix, iy, jx, jy, drawing, img
    jx = x
    jy = y
    vid_file = "./sek2.mp4"
    cap = cv.VideoCapture(vid_file)
    ret, frame = cap.read()
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            img2 = frame
            cv.rectangle(img2, pt1=(ix, iy), pt2=(jx, jy), color=(255, 255, 0), thickness=4)
            img = img2
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        img2 = frame
        cv.rectangle(img2, pt1=(ix, iy), pt2=(jx, jy), color=(255, 255, 0), thickness=4)
        img = img2


cv.namedWindow(winname="Mark the object")
cv.setMouseCallback("Mark the object", draw_reactangle_with_drag)
while True:
    cv.imshow("Mark the object", img)
    if cv.waitKey(10) == 27:
        break
cv.destroyAllWindows()

cap = cv.VideoCapture(vid_file)
_, frame = cap.read()
numFeatures = 20
width, height = 200, 200
cv.waitKey(0)

track_window = (ix, iy, width, height)
roi = frame[iy:iy + height, ix: ix + width]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [20], [0, 180])
plt.hist(roi_hist)
plt.show()
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
plt.hist(roi_hist)
plt.show()
cv.waitKey(0)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        #####Opencv ver
        # _, track_window = cv.meanShift(mask, track_window, term_crit)
        #####Opencv ver
        x, y, w, h = track_window
        print(track_window)
        cor1, cor2 = meanshift_own(mask, x, y)
        track_window = (cor1, cor2, 200, 200)

        print("Track_window: " + str(track_window))
        x, y, w, h = track_window
        final_image = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 3)
        cv.imshow('dst', mask)
        cv.imshow('Mean-Shift', final_image)
        # cv.imshow('Cam-Shift', final_image)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
