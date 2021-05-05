import cv2

cap = cv2.VideoCapture("../dataset/Videos/highway.mp4")

while True:
    ret, frame = cap.read()
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(25)
    if key == 27:
        break
    
cap.release()
cap.destroyAllWindows()