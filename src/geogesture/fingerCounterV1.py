import cv2
import numpy as np



# Initialize video capture
cap = cv2.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)

# ROI information
roi_p1 = (int(width/2), 0)
roi_p2 = (int(width), int(height/2))

#Video loop
while (True):
    # Read frame, flip frame, identify ROI
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, roi_p1, roi_p2, (0,255,0), 2)
    roi_frame = frame[roi_p1[1]:roi_p2[1],roi_p1[0]:roi_p2[0]]
    
    # Create mask and segment hand
    roi_blur = cv2.GaussianBlur(roi_frame, (5,5),0)
    roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0,0,0])
    upper_hsv = np.array([179,255,142])
    mask = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)
    cv2.imshow("Mask", mask)
    
    # Find contour of hand
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        h_contour = max(contours, key = cv2.contourArea) # Assumes biggest area is the hand
        cv2.drawContours(roi_frame, h_contour, -1, (255,0,0), 2)
        h_hull = cv2.convexHull(h_contour)
        cv2.drawContours(roi_frame, [h_hull], -1, (0,0,255), 2, 2)
        # approximate shape of hand
        hand_perim = cv2.arcLength(h_contour,True)
        hand_polygon = cv2.approxPolyDP(h_contour, 0.02*hand_perim, True) # https://medium.com/analytics-vidhya/contours-and-convex-hull-in-opencv-python-d7503f6651bc
        h_pol_hull = cv2.convexHull(hand_polygon,returnPoints=False)
        convex_defects = cv2.convexityDefects(hand_polygon,h_pol_hull)
        if convex_defects is not None:
            count = 0
            points = []
            # print(convex_defects[0])
            for i in range(convex_defects.shape[0]): # for each convex_defects
                start_index, end_index, far_pt_index, fix_dept = convex_defects[i][0] #convex_defects i, [[w,x,y,z]] where each w and x are index of points in hand_poly
                start_pts = hand_polygon[start_index][0]
                end_pts = hand_polygon[end_index][0]
                far_pts = hand_polygon[far_pt_index][0]
                
                # Distance between the start and the end defect point
                c = np.sqrt((end_pts[0] - start_pts[0])**2 + (end_pts[1] - start_pts[1])**2)
                # Distance between the farthest point and the start point
                b = np.sqrt((far_pts[0] - start_pts[0])**2 + (far_pts[1] - start_pts[1])**2)
                # Distance between the farthest point and the end point
                a = np.sqrt((end_pts[0] - far_pts[0])**2 + (end_pts[1] - far_pts[1])**2)
                
                angle = np.arccos((b**2 + a**2 - c**2) / (2*b*a))  # Find each angle
                if angle <= np.pi/2:
                    count += 1
            print(count+1)
            
    
    cv2.imshow("Camera Input", frame)
    cv2.imshow("ROI Input", roi_frame)
    # Check if user wants to exit.
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# When we exit the loop, we have to stop the capture too.
cap.release()
cv2.destroyAllWindows()
