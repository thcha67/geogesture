import cv2
import numpy as np
import json

with open("hsv.json") as f:
    hsv_values = json.load(f)
    h_min, s_min, v_min = hsv_values["hsv_min"]
    h_max, s_max, v_max = hsv_values["hsv_max"]

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", h_min, 179, lambda x: x)
cv2.createTrackbar("Hue Max", "TrackBars", h_max, 179, lambda x: x)
cv2.createTrackbar("Sat Min", "TrackBars", s_min, 255, lambda x: x)
cv2.createTrackbar("Sat Max", "TrackBars", s_max, 255, lambda x: x)
cv2.createTrackbar("Val Min", "TrackBars", v_min, 255, lambda x: x)
cv2.createTrackbar("Val Max", "TrackBars", v_max, 255, lambda x: x)

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = cap.get(3)
height = cap.get(4)

# ROI information
roi_top = int(0)
roi_bot = int(height/2)
roi_left = int(width/2)
roi_right = int(width)
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = img[roi_top:roi_bot,roi_left:roi_right]
    if not success:
        # restart the video
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    # imgStack = stackImages(0.25, ([img, imgHSV], [mask, imgResult]))
    # cv2.imshow("Stacked Images", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("hsv.json", "w") as f:
    json.dump({"hsv_min": [h_min, s_min, v_min], "hsv_max": [h_max, s_max, v_max]}, f)
    
cap.release()
cv2.destroyAllWindows()
