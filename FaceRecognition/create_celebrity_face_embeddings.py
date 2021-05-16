import os
import pandas as pd
import numpy as np
import scipy.io
import time

from PIL import Image

import cv2
import matplotlib.pyplot as plt

import hickle as hkl

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

#-----------------------
haarcascade_frontalface = 'D:/Projects/OpenCV/dataset/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)

# load dlib's CNN face detector
print("Loading CNN face detector...")
MODEL = "D:/Projects/OpenCV/dataset/Weights/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(MODEL)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))
	
	#you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
	from keras.models import model_from_json
	vgg_face_weights = 'D:/Projects/OpenCV/dataset/Weights/vgg_face_weights.h5'
	model.load_weights(vgg_face_weights)
	
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
	
	return vgg_face_descriptor

model = loadVggFaceModel()
print("vgg face model loaded")
 
#------------------------
imdb_data_dir = 'D:/Projects/OpenCV/dataset/imdb_crop'

#download imdb data set here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ . Faces only version (7 GB)
mat = scipy.io.loadmat(imdb_data_dir + '/imdb.mat')
print("imdb.mat meta data file loaded")

columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score", "celeb_names", "celeb_id"]
instances = mat['imdb'][0][0][0].shape[1]
df = pd.DataFrame(index = range(0,instances), columns = columns)

for i in mat:
    if i == "imdb":
        current_array = mat[i][0][0]
        for j in range(len(current_array)):
            #print(j,". ",columns[j],": ",current_array[j][0])
            df[columns[j]] = pd.DataFrame(current_array[j][0])

print("data frame loaded (",df.shape,")")

#-------------------------------

years = ["2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "20015"]

for year in years:
	
	#remove pictures does not include any face
	df = df[df['face_score'] != -np.inf]

	#some pictures include more than one face, remove them
	df = df[df['second_face_score'].isna()]

	#discard inclear ones
	df = df[df['face_score'] >= 5]

	#-------------------------------
	#some speed up tricks. this is not a must.

	#discard old photos
	#df = df[(df['photo_taken'] >= 2005) & (df['photo_taken'] < 2007)]
	df = df[df['photo_taken'] == 2018]

	print("some instances ignored (",df.shape,")")
	#-------------------------------

	def extractNames(name):
		return name[0]

	df['celebrity_name'] = df['name'].apply(extractNames)

	def getImagePixels(image_path):
		#return cv2.imread("imdb_data_set/%s" % image_path[0]) #pixel values in scale of 0-255
		return cv2.imread(imdb_data_dir + "/%s" % image_path[0])

	tic = time.time()
	df['pixels'] = df['full_path'].apply(getImagePixels)
	
	toc = time.time()

	print("reading pixels completed in ",toc-tic," seconds...") #3.4 seconds

	def findFaceRepresentation(img):
		detected_face = img
		
		try: 
			detected_face = cv2.resize(detected_face, (224, 224))
			plt.imshow(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))
			
			#normalize detected face in scale of -1, +1

			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 127.5
			img_pixels -= 1
			
			representation = model.predict(img_pixels)[0,:]
		except:
			representation = None
			
		return representation

	tic = time.time()
	df['face_vector_raw'] = df['pixels'].apply(findFaceRepresentation) #vector for raw image
	toc = time.time()
	print("extracting face vectors completed in ",toc-tic," seconds...")

	tic = time.time()
	df.to_pickle("./representations_2018.pkl")
	# Dump to file
	#hkl.dump(array_obj, 'test.hkl', mode='w')
	toc = time.time()
	print("storing representations completed in ",toc-tic," seconds...")
	time.sleep(1)
	exit(0)


print("data set: ",df.shape)
