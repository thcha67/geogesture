import numpy as np
import matplotlib.pyplot as plt
import cv2

array = np.load("mask.npy").astype(np.uint8)

# array = (255 - array / array.max() * 255).astype(np.uint8)


cv2.namedWindow("Circles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Circles", 640, 480)

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(array)


if len(keypoints) > 1:
    keypoints = sorted(keypoints, key=lambda x: x.size, reverse=True)[:1]

if len(keypoints) == 1:
    print(keypoints[0])
    center = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))
    cv2.circle(array, center, int(keypoints[0].size)//2, (0, 255, 0), 2)
    cv2.circle(array, center, 2, (0, 0, 255), 3)


cv2.imshow("Circles", array)
cv2.waitKey(0)

# array = cv2.drawKeypoints(array, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# cv2.imshow("Circles", array)
# cv2.waitKey(0)

# print(array)

