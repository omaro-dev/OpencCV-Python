import time
import dlib
import cv2

# load dlib's HOG + Linear SVM face detector
print("Loading HOG + Linear SVM face detector...")
hog_face_detector = dlib.get_frontal_face_detector()

# load dlib's CNN face detector
print("Loading CNN face detector...")
MODEL = "D:/Projects/OpenCV/dataset/Weights/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(MODEL)

haarcascade_frontalface = 'D:/Projects/OpenCV/dataset/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)


TEST_VIDEO = "D:/Projects/OpenCV/dataset/Videos/canelo_postfight.mp4"

cap = cv2.VideoCapture(TEST_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second camera: {0}".format(fps))

# Number of frames to capture
num_frames = 1;
print("Capturing {0} frames".format(num_frames))

hog_detections = None
cnn_detections = None

while True:
    # Start time
    start = time.time()
    
    ret, frame = cap.read()
    
    if ret is False:
        break
    
    # Convert video from BGR to RGB channel ordering (which is what dlib expects)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #hog_detections = hog_face_detector(rgb, 1)
    #cnn_detections = cnn_face_detector(rgb, 1)
    cascade_detections = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    if hog_detections:
        for face in hog_detections:
            # ensure the bounding box coordinates fall within the spatial
            # dimensions of the image
            #
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            l = max(0, l)
            t = max(0, t)
            r = min(r, frame.shape[1])
            b = min(b, frame.shape[0])
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            
    elif cnn_detections:
        for face in cnn_detections:
            l, t, r, b = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
            l = max(0, l)
            t = max(0, t)
            r = min(r, frame.shape[1])
            b = min(b, frame.shape[0])
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            
    elif cascade_detections is not None:
        for (x,y,w,h) in cascade_detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Time elapse
    seconds = time.time() - start
    
    # Calculate frames per second
    fps = num_frames / seconds
    cv2.putText(frame, "FPS: " + str(round(fps)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
    
    
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    