from PIL import Image
import cv2
import dlib
import numpy as np
import os
from data_image_loader import get_yale_faces_train_dataset, get_yale_faces_test_dataset

WEIGHTS_DIR = "D:/Projects/OpenCV/dataset/Weights/"

# initialize hog + svm based face detector
face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor(WEIGHTS_DIR + 'shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1(WEIGHTS_DIR + '/dlib_face_recognition_resnet_model_v1.dat')

# Get Train Image
ids, images = get_yale_faces_train_dataset()
#ids, images = get_yale_faces_test_dataset()

# append all face descriptor found
descriptor_matrix = None
train_index = {}
idx = 0

for  id, image in zip(ids, images):
    face_detection = face_detector(image, 1)
    for face in face_detection:
        # l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        # cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)

        points = points_detector(image, face)
        # for point in points.parts():
        #     cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), 1)
            
        # returns a dlib.vector with 128 values for each face
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(image, points)
        #print(type(face_descriptor))
        #print(len(face_descriptor))
    
        # We need to covert this dlib.vector to numpy array and store each face descriptor
        # as a row of a NxM metrics, where M is the descriptor and N is the number of faces
        
        face_descriptor_np = np.array(face_descriptor, dtype=np.float64)
        #print(type(face_descriptor_np))
        #print(len(face_descriptor_np))
        
        # descriptor_matrix
        if descriptor_matrix is None:
            descriptor_matrix = face_descriptor_np
        else:
            descriptor_matrix = np.vstack((descriptor_matrix, face_descriptor_np))
            
        train_index[idx] = {'id': id, 'image':image}
        idx += 1
            
    
print(descriptor_matrix.shape)    

#print(ids)
#print(index)    
#print(index[131].get('id'))   

opencvImage = cv2.cvtColor(np.array(train_index[131].get('image')), cv2.COLOR_RGB2BGR)
cv2.imshow("131", opencvImage)

print(np.linalg.norm(descriptor_matrix[131] - descriptor_matrix[131]))    

print(np.linalg.norm(descriptor_matrix[131] - descriptor_matrix[25])) 
opencvImage = cv2.cvtColor(np.array(train_index[25].get('image')), cv2.COLOR_RGB2BGR)
cv2.imshow("25", opencvImage)

print(np.linalg.norm(descriptor_matrix[131] - descriptor_matrix[130])) 
opencvImage = cv2.cvtColor(np.array(train_index[130].get('image')), cv2.COLOR_RGB2BGR)
cv2.imshow("130", opencvImage)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Validate with test dataset
print("\n======================== Test Validation ================")

test_ids, test_images = get_yale_faces_test_dataset()

threshold = 0.5
min_index = 0
predictions = []
expected_outputs = []

for  id, image in zip(test_ids, test_images):
    face_detection = face_detector(image, 1)
    for face in face_detection:
        points = points_detector(image, face)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(image, points)
        face_descriptor_np = np.array(face_descriptor, dtype=np.float64)
        
        distances = np.linalg.norm(face_descriptor_np - descriptor_matrix, axis = 1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        if min_distance <= threshold:
            print('min_distance = ', min_distance)
            name_pred = train_index[min_index].get('id')
        else:
            name_pred = 'Not identified'
        
        predictions.append(name_pred)
        expected_outputs.append(id)
        
        cv2.putText(image, 'Pred: ' + str(name_pred), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
        cv2.putText(image, 'Exp : ' + str(id), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
    
    testImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    trainImage = cv2.cvtColor(np.array(train_index[min_index].get('image')), cv2.COLOR_RGB2BGR)
    cv2.imshow("Train-Face", trainImage)
    cv2.imshow("Test-Face", testImage)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
exit(0)