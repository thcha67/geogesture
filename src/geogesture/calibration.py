import cv2
import numpy as np
import json
from pathlib import Path

calibration_file = Path("calibration.json")

if calibration_file.exists():
    with open(calibration_file) as f:
        hsv_values = json.load(f)
        h_min, s_min, v_min = hsv_values["min_values"]
        h_max, s_max, v_max = hsv_values["max_values"]
else:
    h_min, s_min, v_min = 0, 0, 0
    h_max, s_max, v_max = 179, 255, 255

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



with open(calibration_file, "w") as f:
    json.dump({"min_values": [h_min, s_min, v_min], "max_values": [h_max, s_max, v_max]}, f)
    print(f"Calibration values saved to {calibration_file.name}")
    
cap.release()
cv2.destroyAllWindows()
