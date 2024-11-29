import cv2
import numpy as np
import json
import scipy.stats
import mouse
# import pyautogui as mouse
# mouse.FAILSAFE = False
from pynput import keyboard
from multiprocessing import Process, Queue, Event
import multiprocessing as mp
from collections import deque
from pathlib import Path


calibration_file = Path("calibration.json")

if not calibration_file.exists():
    raise FileNotFoundError("Calibration file not found. Please run calibration.py first.")

class FingerCounter:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        width, height = self.cap.get(3), self.cap.get(4)
        self.roi_p1 = (int(width/2), 0)
        self.roi_p2 = (int(width), int(height/2))
        self.previous_count = 0
        self.previous_cx = 0
        self.previous_cy = 0
        self.analysis_started = False
        
    def segment_hand(self, frame):
        roi_frame = frame[self.roi_p1[1]:self.roi_p2[1], self.roi_p1[0]:self.roi_p2[0]]
        roi_blur = cv2.GaussianBlur(roi_frame, (5,5),0)
        roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

        with open(calibration_file) as f: # Load calibration data from json file
            calibration = json.load(f)
            min_values = np.array(calibration["min_values"])
            max_values = np.array(calibration["max_values"])
        
        mask = cv2.inRange(roi_hsv, min_values, max_values)
        return mask, roi_frame
    
    def find_contours(self, mask, roi_frame):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            h_contour = max(contours, key = cv2.contourArea) # Assumes biggest area is the hand
            cv2.drawContours(roi_frame, h_contour, -1, (255,0,0), 2)

            h_hull = cv2.convexHull(h_contour)
            cv2.drawContours(roi_frame, [h_hull], -1, (0,0,255), 2, 2)

            return h_contour

        
    def find_fingers(self, h_contour, center_of_mass, roi_frame):
        if h_contour is None:
            return 0
        hand_perim = cv2.arcLength(h_contour, True)
        hand_polygon = cv2.approxPolyDP(h_contour, 0.02*hand_perim, True)
        h_pol_hull = cv2.convexHull(hand_polygon, returnPoints=False)
        h_pol_hull[::-1].sort(axis=0) # Sort hull points in descending order
        convex_defects = cv2.convexityDefects(hand_polygon, h_pol_hull)

        if convex_defects is not None:
            points = hand_polygon[convex_defects[:, 0][:, [0, 1, 2]]][:, :, 0]
            start_pts, end_pts, far_pts = points[:, 0], points[:, 1], points[:, 2]
            
            com_y = center_of_mass[1]

            threshold_line = com_y + 2*(com_y - np.min(far_pts[:, 1]))
            cv2.line(roi_frame, (0, threshold_line), (roi_frame.shape[1], threshold_line), (0, 255, 0), 2)

            for p in far_pts:
                c = 255 if p[1] > threshold_line else 0
                cv2.circle(roi_frame, tuple(p), 5, (0, 0, c), -1)

            dist_start_end = np.linalg.norm(end_pts - start_pts, axis=1)
            dist_far_start = np.linalg.norm(far_pts - start_pts, axis=1)
            dist_far_end = np.linalg.norm(far_pts - end_pts, axis=1)
            
            angles = np.arccos((dist_far_start**2 + dist_far_end**2 - dist_start_end**2) / (2 * dist_far_start * dist_far_end))

            count = np.sum((angles < np.pi / 2) & (far_pts[:, 1] < threshold_line)) + 1
            
            return count if count is not None else 0

    def find_circle(self, mask, h_contour, roi_frame, center_of_mass):
        x, y, w, h = cv2.boundingRect(h_contour)

        new_mask = (mask[y:y+h, x:x+w]).astype(np.uint8)

        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(new_mask)
        
        if len(keypoints) > 1:
            keypoints = sorted(keypoints, key=lambda x: x.size, reverse=True)[:1]
        
        if len(keypoints) == 1:
            center = (int(keypoints[0].pt[0]) + x, int(keypoints[0].pt[1]) + y)
            cv2.circle(roi_frame, center, int(keypoints[0].size), (0, 255, 0), 2)
            cv2.circle(roi_frame, center, 2, (0, 0, 255), 3)
            return center
        
        return None # If no circle is detected


    def run(self, queue, flags):
        center_of_mass = [0,0] 

        while (True):
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            mask, roi_frame = self.segment_hand(frame)
            cv2.rectangle(frame, self.roi_p1, self.roi_p2, (0, 255, 0), 2) # Draw ROI rectangle

            h_contour = self.find_contours(mask, roi_frame) # Find hand contour
            
            if h_contour is not None:
                M = cv2.moments(h_contour)
                try:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center_of_mass = [cx, cy]
                except ZeroDivisionError:
                    pass

                cv2.circle(roi_frame, center_of_mass, 5, (0,255,0), -1)
                count = self.find_fingers(h_contour, center_of_mass, roi_frame)

                circle_center = self.find_circle(mask, h_contour, roi_frame, center_of_mass)
                
                cv2.putText(frame, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA) # Display finger count
                
                if flags[2].is_set(): # Start analysis if p is pressed
                    self.analysis_started = True
                    flags[2].clear()

                if self.analysis_started:
                    queue.put((count, circle_center))
                    flags[0].set()

            cv2.imshow("Frame", frame) # Display frame
            cv2.waitKey(1) # Needed to display frame

            if flags[1].is_set():
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def process(queue, flags):
    fc = FingerCounter(0)
    fc.run(queue, flags)

def keyboard_listener(flags):
    def on_press(key):
        try:
            if key.char == "q":
                flags[1].set()
                return False
            elif key.char == "p":
                flags[2].set()
        except AttributeError:
            pass
        
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    green_bos_corners = np.array([[55, 55], [260, 180]])
    screen_corners = np.array([[0,0], [1920, 1080]])

    x_factor, x_offset = np.polyfit(green_bos_corners[:,0], screen_corners[:,0], 1)
    y_factor, y_offset = np.polyfit(green_bos_corners[:,1], screen_corners[:,1], 1)


    mp.freeze_support()
    queue = Queue()
    flags = [Event(), Event(), Event()] # flags[0] for new data, flags[1] for exit, flags[2] for analysis start
    process_task = Process(target=process, args=(queue, flags))
    keyboard_task = Process(target=keyboard_listener, args=(flags,))

    process_task.start()
    keyboard_task.start()

    buffer_length = 10

    buffer_count = deque(maxlen=buffer_length)  # Buffer to store the last 5 finger counts
    buffer_count.extend([2]*buffer_length) # Initialize buffer with 5s (release)

    buffer_is_circle = deque(maxlen=buffer_length)  # Buffer to store the last 5 is_circle values
    buffer_is_circle.extend([False]*buffer_length) # Initialize buffer with False


    previous_center = None

    while True:    
        if flags[0].is_set() and not flags[2].is_set():
            
            count, center = queue.get()
            
            buffer_count.append(count)
            count = max(set(buffer_count), key=buffer_count.count) # Get the most common finger count in the buffer

            buffer_is_circle.append(center is not None)

            if any(buffer_is_circle): # If any of the last 5 frames is a circle, then it is a circle since the circle detection almost never has false positives
                if center is None:
                    center = previous_center
            
            if center:
                x, y = center
                x = int(x * x_factor + x_offset)
                y = int(y * y_factor + y_offset)

                print(x, y, sum((x, y)), sum(screen_corners[1]))
                
                if abs(sum((x, y))) > sum(screen_corners[1]): # check if the center has a valid value
                    continue # Skip this iteration if the center is invalid

                if count in (2, 3, 4, 5):

                    mouse.drag(*(previous_center or center), x, y)
                else:
                    mouse.move(x, y)
                previous_center = (x, y)

            else:
                if count == 1 or count is None: # Click
                    print("Click")
                    mouse.click()
                elif count == 3:  # Scroll up
                    #print("Scroll up")
                    mouse.wheel(1)
                elif count == 5:  # Scroll down
                    #print("Scroll down")
                    mouse.wheel(-1)
            
            flags[0].clear()

        if flags[1].is_set():
            break
    
    process_task.join()
    keyboard_task.terminate()

