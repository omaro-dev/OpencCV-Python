import cv2
import dlib
import time

print('dlib.DLIB_USE_CUDA = ', dlib.DLIB_USE_CUDA)

# initialize hog + svm based face detector
face_detector_hog = dlib.get_frontal_face_detector()

# initialize face point detection
points_detector = dlib.shape_predictor('D:/Projects/OpenCV/dataset/Weights/shape_predictor_68_face_landmarks.dat')

# Draw point detection on face(s) passed in
def draw_face_point_detection(image, faces_detected):
    for face in faces_detected:
        points = points_detector(image, face)
        for point in points.parts():
            cv2.circle(image, (point.x, point.y), 2, (255,0,0), 1)

    return image
    
    
IMAGE_DIR = "D:/Projects/OpenCV/dataset/Images/"

image = cv2.imread(IMAGE_DIR + "People2.jpg")
print("Orig Shape:", image.shape)
# cv2.imshow("People-2", image)

# Resize Image
# image = cv2.resize(image, (int(image.shape[1]*0.8), int(image.shape[0]*0.8)))
image = cv2.resize(image, (800, 600))
print("New Shape: ", image.shape)
# cv2.imshow("People-2", image)


start = time.time()

# We don't need to pass a grayscale image to this classifier.
# Because it used Histogram of Oriented Gradient algorithm, results are better if images are BGR
# Param 1 is the number of times it should upsample the image. By default, 1 works for most cases. 
# (Upsampling the image helps to detect smaller faces)
# 
detections = face_detector_hog(image, 1) 
#draw_face_point_detection(image, detections)

end = time.time()
print("Execution Time (in seconds) : ", format(end - start, '.2f'))

print(len(detections))

# Rectangle format in dlib and OpenCV are a bit different. The face_detector_hog() returns 
# rectangles with values [left(), top(), right(), bottom()]
for face in detections:
    # print(face.left())
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)
    
    # Or converting to Opencv format
    #x = face.left()
    #y = face.top()
    #w = face.right() - x
    #h = face.bottom() - y
    #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
     
cv2.imshow("Faces People-2", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
