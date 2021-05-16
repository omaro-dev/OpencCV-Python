import numpy as np
from PIL import Image
from numpy.lib.type_check import imag
import cv2
import dlib
import os

# Yale Faces Dataset
YALE_FACES_TRAIN_DIR = 'D:/Projects/OpenCV/dataset/yalefaces/train'
YALE_FACES_TEST_DIR = 'D:/Projects/OpenCV/dataset/yalefaces/test'
#print(os.listdir(YALE_FACES_TRAIN_DIR))


def get_yale_faces_train_dataset():
    paths = [os.path.join(YALE_FACES_TRAIN_DIR, f) for f in os.listdir(YALE_FACES_TRAIN_DIR)]
    #print(paths)
    faces = []
    ids = []
    for path in paths:
        #print(path)
        image = Image.open(path).convert('RGB')
        #print(type(image))
        image_np = np.array(image, 'uint8')
        #print(type(image_np))
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        #print(id)
        ids.append(id)
        faces.append(image_np)
    return np.array(ids), faces


def get_yale_faces_test_dataset():
    paths = [os.path.join(YALE_FACES_TEST_DIR, f) for f in os.listdir(YALE_FACES_TEST_DIR)]
    #print(paths)
    faces = []
    ids = []
    for path in paths:
        #print(path)
        image = Image.open(path).convert('RGB')
        #print(type(image))
        image_np = np.array(image, 'uint8')
        #print(type(image_np))
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        #print(id)
        ids.append(id)
        faces.append(image_np)
    return np.array(ids), faces


# CalTech Faces Dataset
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CALTECH_FACES_TRAIN_DIR = "D:/Projects/OpenCV/PyImageSearch_TutorialsAndCode/face-reco-lbps/caltech_faces/train"
CALTECH_FACES_TEST_DIR = "D:/Projects/OpenCV/PyImageSearch_TutorialsAndCode/face-reco-lbps/caltech_faces/test"

face_detector = dlib.get_frontal_face_detector()

def load_face_dataset(image_dir, validateFaces=False):
	# grab the paths to all images in our input directory, extract
	# the name of the person (i.e., class label) from the directory
	# structure, and count the number of example images we have per
	# face
    imagePaths = []
    for (rootDir, dirNames, filenames) in os.walk(image_dir):
        # loop over the filenames in the current directory
        for filename in filenames:
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(image_types):
                # construct the path to the image and yield it
                path = os.path.join(rootDir, filename)
                imagePaths.append(path)
 
    #print(len(imagePaths))
    #print(imagePaths)
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    #print(names, counts)
    names = names.tolist()

	# initialize lists to store our extracted faces and associated
	# labels
    images = []
    labels = []

    # loop over the image paths
    for i, imagePath in enumerate(imagePaths):
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]
                
        if validateFaces:
            (height, width) = image.shape[:2]
            if width < 300 or height < 300:
                #print("path[", i, "]: ", imagePath, " Orig shape: ", image.shape)
                image = cv2.resize(image, (320,320), interpolation=cv2.INTER_AREA)
                #print("path[", i, "]: ", imagePath, " New shape: ", image.shape)
                
            #print("path[", i, "]: ", imagePath, " shape: ", image.shape)
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_detection = face_detector(rgb, 1)
            # Make sure a face can be detected
            if len(face_detection) < 1:
                print("No Face found....path: ", imagePath)
                continue
        
        # if i == 0:
        #     images = np.expand_dims(np.array(image, dtype= float)/255, axis= 0)
        # else:
        #     image_np = np.expand_dims(np.array(image, dtype= float)/255, axis= 0)
        #     images= np.append(images, image_np, axis= 0)
      
        images.append(image)
        labels.append(name)

	# convert our images and labels lists to NumPy arrays
    # images = np.array(images)
    labels = np.array(labels)

    # return a 2-tuple of the faces and labels
    return (images, labels)

def load_caltech_face_train_dataset():
    return load_face_dataset(CALTECH_FACES_TRAIN_DIR)


def load_caltech_face_test_dataset():
    return load_face_dataset(CALTECH_FACES_TEST_DIR)

def load_family_faces(validateFaces=False):
    images_dir = "C:/Users/ocamp/Pictures/Faces"
    return load_face_dataset(images_dir, validateFaces)

# images, labels = load_family_faces(True)
# print(len(images))
# print(len(labels))
# for img in images: 
#     print("Image shape: ", img.shape)

# ids, faces = get_yale_faces_test_dataset()
# #ids, faces = get_yale_faces_train_dataset()
# print(len(ids))
# print(len(faces))
# print(ids[0])
# print(faces[0].shape)

