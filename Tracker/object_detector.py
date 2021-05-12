import cv2
import numpy as np
import time
from object_tracker import EuclideanDistTracker

cap = cv2.VideoCapture("../dataset/Videos/highway.mp4")

# Create tracker object
tracker = EuclideanDistTracker()

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# For FPS
prev_frame_time = 0 # used to record the time when we processed last frame
new_frame_time = 0 # used to record the time at which we processed current frame

while True:
    ret, frame = cap.read()
    
    # if video finished or no Video Input
    if not ret:
        break
        
    height, width, _ = frame.shape
    
    # Extract Region of interest
    roi = frame[340:720, 500:800]
    
    # 1. Object Detection
    mask = object_detector.apply(roi)
    _ , mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    # time when we finish processing for this frame
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    # puting the FPS count on the frame
    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
  
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("roi", roi)
    
    # key = cv2.waitKey(25)
    # if key == 27:
    #     break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cap.destroyAllWindows()