import cv2

## CALLBACK FUNCTIONS
def draw_rectangle(event, x, y, flags, params):
    global pt1, pt2, topLeft_clicked, topRight_clicked 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Reset the Rectangle
        if topLeft_clicked == True and topRight_clicked == True:
            pt1 = (0, 0)
            pt2 = (0, 0)
            topLeft_clicked = False
            topRight_clicked = False
            return
        
        if topLeft_clicked == False:
            pt1 = (x, y)
            topLeft_clicked = True
            
        elif topRight_clicked == False:
            pt2 = (x, y)
            topRight_clicked = True
    
    
## GLOBAL VARIABLES
pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
topRight_clicked = False

## CONNECT TO THE CALLBACK
cap = cv2.VideoCapture(0)
cv2.namedWindow('MyWeb-Cam')
cv2.setMouseCallback('MyWeb-Cam', draw_rectangle)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Drawing on the frame based off the global variables
    if topLeft_clicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0, 0, 255), thickness=-1)
        
    if topLeft_clicked and topRight_clicked:
        cv2.rectangle(frame, pt1, pt2, (0,0,255), thickness=3)
        
    cv2.imshow('MyWeb-Cam', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
