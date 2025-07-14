
import cv2
#from matplotlib import pyplot
import numpy as np
from PIL import Image

#from sklearn.metrics import accuracy_score
#import pandas as pd 
from mtcnn.mtcnn import MTCNN
#from matplotlib import pyplot
#from keras.models import load_model
#from PIL import Image
#from keras import engine,utils
#from keras_applications import vgg16
import os
import glob
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

# extract a single face from a given photograph
def extract_face(image, required_size=(224, 224)):
    # load image from file
    pixels = cv2.imread(image)
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
        temp=cv2.rectangle(pixels,(x1,y1),(x2,y2),(0,0,255),2)
        #cv2.imwrite('C:/Users/u27c79/Documents/Project/Face_recongnition_System/images/bound/{}.jpg'.format(filename),temp)
        # extract the face
        face = pixels[y1:y2,x1:x2 ]
        #cv2.imshow("test",face)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        print(face.shape)
        face2=cv2.resize(face,required_size)
        # resize pixels to the model size
        #image = Image.fromarray(face)
        #mage = image.resize(required_size)
        return face2
    
# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(faces):
    # extract faces
    #faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # samples=faces
    #print(samples.shape)
    samples = np.expand_dims(samples, axis=0)
    #print(samples.shape)

    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    #print(samples.shape)

    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
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

cam = cv2.VideoCapture("https://192.168.1.78:8080/video")

cv2.namedWindow("test")

img_counter = 0

while True:
    cam.set(cv2.CAP_PROP_BUFFERSIZE,3)
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    else:
        img_name = "imagess_{}.jpg".format(img_counter)
        cv2.imwrite("C://Users//u27c79//Documents//Project//Face_recongnition_System//image_face/{}".format(img_name), frame)
        print("{} capture!".format(img_name))
        try:
            face=extract_face('C://Users//u27c79//Documents//Project//Face_recongnition_System//image_face/{}'.format(img_name))
            #if face is None:
                #pass
            #else:
            # for i in face:
            embeddings=get_embeddings(face)
        #embeddings=get_embeddings(face)
            print([is_match(base_embedding,embeddings),img_name],"\n")
        except:
            pass    
    img_counter += 1

cam.release()

cv2.destroyAllWindows()


#flag=0
#faces = [extract_face(f) for f in filenames]
"""for i in faces:
    if i is not None:
        embeddings=get_embeddings(i)
        output.append([is_match(div_embedding,embeddings),filenames[flag]])
    flag+=1
    
for i in output:
    print(i,"\n") """       