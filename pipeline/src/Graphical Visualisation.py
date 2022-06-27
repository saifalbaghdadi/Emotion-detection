import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FacialExpressionModel

# Creating an instance of the class with the parameters as model and its weights
test_model = FacialExpressionModel("../model/model_json.json" , "../model/model_weights_s1.h5")

''' Here I using the face detection function using Haar Cascades.
It is used for object detection using cascading classifiers based on Haar feature and is an effective method for object detection.
You can find the list of haarcascade XML files from this link (https://github.com/Itseez/opencv/tree/master/data/haarcascades). '''

# Loading the classifier from the file.
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# It does prediction of Emotions found in the Image provided, does the Graphical visualisation, saves as Images and returns them.
def Emotion_Analysis(img):

    # Read the Image through OpenCv's imread()
    path = "static/" + str(img)
    image = cv2.imread(path)

    # Convert the Image into Gray Scale
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image size is reduced by 30% at each image scale.
    scaleFactor = 1.3

    # 5 neighbors should be present for each rectangle to be retained.
    minNeighbors = 5

    # Detect the Faces in the given Image and store it in faces.
    faces = facec.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

    # When Classifier could not detect any Face.
    if len(faces) == 0:
        return [img]

    