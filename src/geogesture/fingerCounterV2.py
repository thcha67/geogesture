import cv2
import numpy as np
import json
import mouse
import keyboard
from multiprocessing import Process, Queue, Event
import multiprocessing as mp


class FingerCounter:
    
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        self.width = self.cap.get(3)
        self.height = self.cap.get(4)
        self.roi_p1 = (int(self.width/2), 0)
        self.roi_p2 = (int(self.width), int(self.height/2))
        self.previous_count = 0
        self.previous_cx = 0
        self.previous_cy = 0
        
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
            cv2.drawContours(roi_frame, [h_hull], -1, (0,0,255), 2, 2)

    
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
            
            
    def run(self, queue, flags):
        center_of_mass = [0,0]
        i = 0
        speed_each_n_frames = 5
        while (True):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            mask, roi_frame = self.segment_hand(frame)
            cv2.rectangle(frame, self.roi_p1, self.roi_p2, (0, 255, 0), 2)
            # cv2.imshow("Mask", mask)
            h_contour = self.find_contours(mask, roi_frame)
            if h_contour is not None and i % speed_each_n_frames == 0:
                M = cv2.moments(h_contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                center_of_mass = [cx, cy]
            cv2.circle(roi_frame, center_of_mass, 5, (0,255,0), -1)

            count = self.find_fingers(h_contour)
            cv2.putText(frame, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Frame", frame)
            i += 1
            self.analyse([count, center_of_mass], queue, flags[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                flags[1].set()
                break
                

            
        self.cap.release()
        cv2.destroyAllWindows()


    def analyse(self, returns, queue, flag):
        count = returns[0]
        cx, cy = returns[1]
        # check for new data
        if (count != self.previous_count) or (cx != self.previous_cx) or (cy != self.previous_cy):
            queue.put(returns)
            self.previous_count = count
            self.previous_cx = cx
            self.previous_cy = cy
            flag.set()
        

def process(queue, flags):
    fc = FingerCounter(0)
    fc.run(queue, flags)

if __name__ == "__main__":
    mp.freeze_support()
    queue = Queue()
    flags = [Event(), Event()]
    p = Process(target=process, args=(queue, flags))
    p.start()
    while True:
        if flags[0].is_set():
            count, center_of_mass = queue.get()
            print(count, center_of_mass)
            flags[0].clear()
        if flags[1].is_set():
            break
    p.join()
    
    
        
