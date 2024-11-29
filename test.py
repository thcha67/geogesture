import numpy as np
import matplotlib.pyplot as plt
import cv2

array = np.load("mask.npy").astype(np.uint8)

array = (255 - array / array.max() * 255).astype(np.uint8)
#array = cv2.GaussianBlur(array, (9, 9), 0)

# choose cv2 window size
cv2.namedWindow("Circles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Circles", 640, 480)

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(array)

array = cv2.drawKeypoints(array, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow("Circles", array)
cv2.waitKey(0)

print(array)

