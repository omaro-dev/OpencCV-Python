import cv2
import os
import time

IMAGE_DIR = "D:/Projects/OpenCV/dataset/Images/"
CASCADE_DIR = "D:/Projects/OpenCV/dataset/haarcascades/"

if os.path.exists(IMAGE_DIR + "People1.jpg"):
    if os.path.isfile(IMAGE_DIR + "People1.jpg"):
        print("File: " + IMAGE_DIR + "People1.jpg")
    elif os.path.isdir(IMAGE_DIR + "People1.jpg"):
        print("Dir: " + IMAGE_DIR + "People1.jpg")
else:
    print('File does NOT exist - ' + IMAGE_DIR + "People1.jpg")
    
image = cv2.imread(IMAGE_DIR + "People1.jpg")
print("Orig Shape:", image.shape)

# Resize Image
image = cv2.resize(image, (800, 600))
print("New Shape: ", image.shape)

# Convert to Grayscale - Cascade Classifier recommend to work on gray images
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray Image", image_gray)

# Detecting faces
face_detector = cv2.CascadeClassifier(CASCADE_DIR + '/haarcascade_frontalface_default.xml')

start = time.time()

# A small scaleFactor can be used when we have smaller faces
# Similarly, the larger the faces, the larger the scaleFactor value
# This parameter must be obtained by testing our specific application as it depends on
# the size/quality of the image 
detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.19)

end = time.time()
print("Execution Time (in seconds) : ", format(end - start, '.2f'))

#print(detections)
print("Number of Faces in People1 image = ", len(detections))
for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), thickness=3)
    
cv2.imshow("People-1", image)

# Find faces on 2nd image
image2 = cv2.imread(IMAGE_DIR + "People2.jpg")
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
print("Orig Shape:", image2_gray.shape)

start = time.time()

# minNeighbors Parameter specifying how many neighbors (i.e. other bounding boxes) each 
# candidate rectangle should have
# the default value of minSize is (30,30)
detections = face_detector.detectMultiScale(image2_gray, scaleFactor=1.2, minNeighbors=7)

end = time.time()
print("Execution Time (in seconds) : ", format(end - start, '.2f'))

print("Number of Faces in People2 image = ", len(detections))
for (x, y, w, h) in detections:
    cv2.rectangle(image2, (x, y), (x+w, y+h), (0,255,0), thickness=3)

cv2.imshow("People-2", image2)

cv2.waitKey(0)
cv2.destroyAllWindows()