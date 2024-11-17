import cv2
import numpy as np
import json


class FingerCounter:
    
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        self.width = self.cap.get(3)
        self.height = self.cap.get(4)
        self.roi_p1 = (int(self.width/2), 0)
        self.roi_p2 = (int(self.width), int(self.height/2))
        
    def segment_hand(self, frame):
        roi_frame = frame[self.roi_p1[1]:self.roi_p2[1],self.roi_p1[0]:self.roi_p2[0]]
        roi_blur = cv2.GaussianBlur(roi_frame, (5,5),0)
        roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

        with open("hsv.json") as f:
            hsv_values = json.load(f)
            lower_hsv = np.array(hsv_values["hsv_min"])
            upper_hsv = np.array(hsv_values["hsv_max"])
        
        mask = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)
        return mask, roi_frame
    
    def find_contours(self, mask, roi_frame):
        contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            h_contour = max(contours, key = cv2.contourArea) # Assumes biggest area is the hand
            cv2.drawContours(roi_frame, h_contour, -1, (255,0,0), 2)
            h_hull = cv2.convexHull(h_contour)
            h_hull[::-1].sort(axis=0)
            cv2.drawContours(roi_frame, [h_hull], -1, (0,0,255), 2, 2)
            # find center of mass
            M = cv2.moments(h_contour)
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(roi_frame, (cx, cy), 5, (0,255,0), -1)
            except ZeroDivisionError:
                pass
    
            return h_contour
        else:
            return None

        
    def find_fingers(self, h_contour):
        if h_contour is None:
            return 0
        hand_perim = cv2.arcLength(h_contour,True)
        hand_polygon = cv2.approxPolyDP(h_contour, 0.02*hand_perim, True)
        h_pol_hull = cv2.convexHull(hand_polygon, returnPoints=False)
        h_pol_hull[::-1].sort(axis=0)
        convex_defects = cv2.convexityDefects(hand_polygon, h_pol_hull)
        if convex_defects is not None:
            count = 0
            points = []
            start_indexs = convex_defects[:,0][:,0]
            end_indexs = convex_defects[:,0][:,1]
            far_pts_indexs = convex_defects[:,0][:,2]
            start_pts = hand_polygon[start_indexs][:,0]
            end_pts = hand_polygon[end_indexs][:,0]
            far_pts = hand_polygon[far_pts_indexs][:,0]
            dist_start_end = np.sqrt((end_pts[:,0] - start_pts[:,0])**2 + (end_pts[:,1] - start_pts[:,1])**2)
            dist_far_start = np.sqrt((far_pts[:,0] - start_pts[:,0])**2 + (far_pts[:,1] - start_pts[:,1])**2)
            dist_far_end = np.sqrt((end_pts[:,0] - far_pts[:,0])**2 + (end_pts[:,1] - far_pts[:,1])**2)
            angles = np.arccos((dist_far_start**2 + dist_far_end**2 - dist_start_end**2) / (2*dist_far_start*dist_far_end))
            count = np.sum(angles < np.pi/2) + 1
            if count is None:
                return 0
            return count
            
            
    def run(self):
        while (True):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            mask, roi_frame = self.segment_hand(frame)
            cv2.rectangle(frame, self.roi_p1, self.roi_p2, (0,255,0), 2)
            cv2.imshow("Mask", mask)
            h_contour = self.find_contours(mask, roi_frame)
            count = self.find_fingers(h_contour)
            cv2.putText(frame, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    fc = FingerCounter(0)
    fc.run()
