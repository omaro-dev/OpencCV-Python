import cv2
import dlib
import numpy as np
import os
from data_image_loader import load_caltech_face_train_dataset, load_caltech_face_test_dataset


WEIGHTS_DIR = "D:/Projects/OpenCV/dataset/Weights/"

# initialize hog + svm based face detector
face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor(WEIGHTS_DIR + 'shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1(WEIGHTS_DIR + '/dlib_face_recognition_resnet_model_v1.dat')

# Get Dataset Image
train_images, train_labels = load_caltech_face_train_dataset()
print("[INFO] {} images in train dataset".format(len(train_images)))

# append all face descriptor found
descriptor_matrix = None
faces_not_found = 0
show_image = False
train_index = {}
idx = 0

for  label, image in zip(train_labels, train_images):
    # Convert video from BGR to RGB channel ordering (which is what dlib expects)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_detection = face_detector(rgb, 1)
    # For face recognition we need only one face, let's calculate the face area reported
    #
    if len(face_detection) < 1:
        faces_not_found += 1
        continue
        
    # if len(face_detection) > 1:
    #     print("Found more than one face: ", len(face_detection))
    #     show_image = True
    #     for face in face_detection:
    #         l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    #         width = r - l
    #         length = b - t
    #         area = width * length
    #         print("rect area = ", area)
            
    for face in face_detection:
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        # Calculate rect area to see if it's a valid face
        # it small face/ares (e.g. < 11000) skip face
        area = (r - l) * (b - t)
        if area < 11000:
            continue
        
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)
        cv2.putText(image, 'Name: ' + str(label), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
        
        points = points_detector(rgb, face)
        # for point in points.parts():
        #     cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), 1)
            
        # returns a dlib.vector with 128 values for each face
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(rgb, points)
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
            
        train_index[idx] = {'id': label, 'image':image}
        idx += 1
        
    # if show_image:
    #     cv2.imshow("Faces", image)
    #     cv2.waitKey(0)
    #     show_image = False
            
print("Number of faces NOT found in Image: ", faces_not_found)
print("train_index size: ", len(train_index))
print("descriptor_matrix size: ", descriptor_matrix.shape)    

# Validate with test dataset
print("\n======================== Test Validation ================")

test_images, test_labels = load_caltech_face_test_dataset()
print("[INFO] {} images in test dataset".format(len(test_images)))

threshold = 0.5
min_index = 0
predictions = []
expected_outputs = []

for  id, image in zip(test_labels, test_images):
    # Convert video from BGR to RGB channel ordering (which is what dlib expects)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_detection = face_detector(rgb, 1)
    
    if len(face_detection) < 1:
        print("WARN: face NOT found in test image")
        continue
    
    if len(face_detection) > 1:
        print("Test image has more than one face, ", len(face_detection))
        # cv2.imshow("Many Faces", image)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Many Faces")
        
    for face in face_detection:    
        # Calculate rect area to see if it's a valid face
        # if it's a small face/ares (e.g. < 2500) skip face
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        area = (r - l) * (b - t)
        print("react area = ", area)
        if area < 11000:
            continue
                
        points = points_detector(rgb, face)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(rgb, points)
        face_descriptor_np = np.array(face_descriptor, dtype=np.float64)
        
        distances = np.linalg.norm(face_descriptor_np - descriptor_matrix, axis = 1)
        #print("distances len = ", len(distances))
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
        
    cv2.imshow("Train-Face", train_index[min_index].get('image'))
    cv2.imshow("Test-Face", image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
