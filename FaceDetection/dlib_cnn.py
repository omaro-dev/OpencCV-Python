import cv2
import dlib 
import time

print('dlib.DLIB_USE_CUDA = ', dlib.DLIB_USE_CUDA)

IMAGE_DIR = "D:/Projects/OpenCV/dataset/Images/"
WEIGHTS_DIR = "D:/Projects/OpenCV/dataset/Weights/"

image = cv2.imread(IMAGE_DIR + "People1.jpg")
print("Orig Shape:", image.shape)
#cv2.imshow("People-2", image)


# Resize Image
# image = cv2.resize(image, (int(image.shape[1]*0.8), int(image.shape[0]*0.8)))
image = cv2.resize(image, (800, 600))
print("New Shape: ", image.shape)
# cv2.imshow("People-2", image)

# initialize cnn based face detector with the weights
cnn_detector = dlib.cnn_face_detection_model_v1(WEIGHTS_DIR + 'mmod_human_face_detector.dat')

start = time.time()

detections = cnn_detector(image, 1) 

end = time.time()
print("Execution Time (in seconds) : ", format(end - start, '.2f'))

print(len(detections))

# Rectangle format in dlib and OpenCV are a bit different. The face_detector_hog() returns 
# rectangles with values [left(), top(), right(), bottom()]
for face in detections:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    #print("Confidence ", face.confidence)

cv2.imshow("Faces People-2", image)

cv2.waitKey(0)
cv2.destroyAllWindows()