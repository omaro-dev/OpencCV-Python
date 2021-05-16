import os
import glob
import pandas as pd
import numpy as np
import scipy.io
import time
import cv2
import dlib
from data_image_loader import load_family_faces

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


def loadProcessedData():
	tic = time.time()
	dfs = []
	for filename in glob.glob('D:/Projects/OpenCV/dataset/celebrity_faces_pkl/*.pkl'):
		print("pickle file = ", filename)
		newDF = pd.read_pickle(filename)
		dfs.append(newDF)

	new_df = pd.concat(dfs, ignore_index=True)
	toc = time.time()
	print("reading pre-processed data frame completed in ",toc-tic," seconds...")
 
	return new_df


face_detector = dlib.get_frontal_face_detector()

def detectFace(src_image, label, show_face:False):
	#faces = face_cascade.detectMultiScale(my_image, 1.3, 5)
	rgb = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
	face_detections = face_detector(rgb, 1)
    
	if len(face_detections) < 1:
		print("WARN: could NOT detect face ")
		return None
	elif len(face_detections) > 1:
		print("WARN: more then One face detected.....expecting a single face")
		return None

	for face in face_detections:
		x = face.left()
		y = face.top()
		w = face.right() - x
		h = face.bottom() - y
		detected_face = src_image[int(y):int(y+h), int(x):int(x+w)] #crop detected face

		#add 5% margin around the face
		try:
			resolution_x = src_image.shape[1]
			resolution_y = src_image.shape[0]
			margin = 0.05 #5
			margin_x = int((w * margin)/100)
			margin_y = int((h * margin)/100)
			if y-margin_y > 0 and x-margin_x > 0 and y+h+margin_y < resolution_y and x+w+margin_x < resolution_x:
				detected_face = src_image[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
		except:
			print("detected face has no margin")
     			
	if show_face:
		cv2.rectangle(src_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
		cv2.putText(src_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
		cv2.imshow("Face Detected", src_image)
		cv2.waitKey(0)
		cv2.destroyWindow("Face Detected")
	
	return detected_face
   

def findCelebrityFace(detected_face, df, show_face:False):    
	# Directory where Celebritites faces are stored
	CELEB_FACES_DIR = 'D:/Projects/OpenCV/dataset/imdb_crop'
 
	detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
	img_pixels = image.img_to_array(detected_face)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	#normalize in scale of [-1, +1]
	img_pixels /= 127.5
	img_pixels -= 1
			
	captured_representation = model.predict(img_pixels)[0,:]	
 
	def findCosineSimilarity(source_representation, test_representation=captured_representation):
		try:
			a = np.matmul(np.transpose(source_representation), test_representation)
			b = np.sum(np.multiply(source_representation, source_representation))
			c = np.sum(np.multiply(test_representation, test_representation))
			return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
		except:
			return 10 #assign a large value. similar faces will have small value.

	df['similarity'] = df['face_vector_raw'].apply(findCosineSimilarity)
 
	#look-alike celebrity
	min_index = df[['similarity']].idxmin()[0]
	instance = df[df.index == min_index]
	name = instance['celebrity_name'].values[0]
	similarity = instance['similarity'].values[0]
	similarity = (1 - similarity)*100

	print("\n-----------------")
	print(name," (",similarity," %)")
	print("-----------------")

	if similarity > 50:
		file_name = str(instance['full_path'].values[0])
		file_name = file_name.replace('[\'','')
		file_name = file_name.replace('\']','')
		full_path = CELEB_FACES_DIR + "/" + str(file_name)    
		#print("full_path = ", full_path)

		celebrity_img = cv2.imread(full_path)
  
		if show_face:
			cv2.putText(celebrity_img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
			cv2.imshow("Celeb Face", celebrity_img)
			cv2.waitKey(0)
			cv2.destroyWindow("Celeb Face")
   
		return (name, similarity, celebrity_img)

	else:
		print("WARN: no celebrity face match ")
		return (None, None, None)
      
      
#--------------------------------------------

print("Loading vgg face model...")
model = loadVggFaceModel()
print("vgg face model loaded")

print("Loading pre-processed files....")
df = loadProcessedData()
print("pre-processed files loaded....data set: ", df.shape)
	

images, labels = load_family_faces(True)
print("Number of Images to process: ", len(images))

for my_label, my_image in zip(labels, images):
	# Check image has a minimum size
	resolution_x = my_image.shape[1]
	resolution_y = my_image.shape[0]
 
	if resolution_x < 300 or resolution_y < 300:
		print("Image too small (", my_image.shape, ")....skipping it ")
		# cv2.imshow("Small Image", my_image)
		# cv2.waitKey(0)
		# cv2.destroyWindow("Small Image")
		continue
    
	subject_face = detectFace(my_image, my_label, False)
	if subject_face is None:
		print("Program could NOT detect ", my_label, " face....skipping it")
		continue

	# Convert video from BGR to RGB channel ordering (which is what dlib expects)
	#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	celeb_name, similarity, celeb_image = findCelebrityFace(subject_face, df, False)
	if celeb_name is None:
		print("Nothing to do")
		continue

	# Render Subject Image with Celeb Face
	celebrity_img = cv2.resize(celeb_image, (112, 112))
				
	try:	
		my_image[10:122, resolution_x-112:resolution_x] = celebrity_img
		label = celeb_name + " (" + "{0:.2f}".format(similarity) + "%)"
		cv2.putText(my_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
		# connect face and text
		#face_x = subject_face.shape[1]
		#face_y = subject_face.shape[0]
		# cv2.line(my_image,(resolution_x-face_x, face_y),(x+w-25, y-64),(67,67,67),1)
		# cv2.line(my_image,(int(x+w/2),y),(x+w-25,y-64),(67,67,67),1)
	except Exception as e:
		print("exception occured: ", str(e))
 
	cv2.imshow('Image', my_image)
	key = cv2.waitKey(0)		
	if key & 0xFF == ord('q'): #press q to quit
		break
	cv2.destroyAllWindows()

