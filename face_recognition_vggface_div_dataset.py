import cv2
#from matplotlib import pyplot
import numpy as np
#from PIL import Image

#from sklearn.metrics import accuracy_score
#import pandas as pd 
from mtcnn.mtcnn import MTCNN
#from matplotlib import pyplot
#from keras.models import load_model
#from PIL import Image
#from keras import engine,utils
#from keras_applications import vgg16
#import os
import glob
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = cv2.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the imagepip 
    results = detector.detect_faces(pixels)
    #print(filename)
    print(len(results)>0)
    if (len(results)>0):
    # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2,x1:x2 ]
        print(face.shape)
        face2=cv2.resize(face,required_size)
        return face2
    
#calculate face embeddings for a list of photo files
def get_embeddings(faces):
    # extract faces
    #faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    #samples = np.expand_dims(samples, axis=0)
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    #print(model.summary())
    # perform prediction
    yhat = model.predict(samples)
    return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.4):
    # calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		return 'face is Match and score={}'.format(score)
	else:
		return 'face is NOT Match and score={}'.format(score)

folder='C:/Users/u27c79/Documents/Project/Face_recongnition_System/images/*.jpg'
filenames=glob.glob(folder)

base_face=extract_face('C:/Users/u27c79/Documents/Project/Face_recongnition_System/image_dataset/image_0.jpg')
base_embedding=get_embeddings(base_face)

output=[]
flag=0
faces = [extract_face(f) for f in filenames]
for i in faces:
    if i is not None:
        embeddings=get_embeddings(i)
        output.append([is_match(base_embedding,embeddings),filenames[flag]])
    flag+=1
    
for i in output:
    print(i,"\n")        