# Emotion Detection Project
- Emotion Detection has always been an easy task for humans, but achieving the same task with a computer algorithm is quite challenging. With the recent advancement in computer vision and machine learning, it is possible to detect emotions from images.
In this project, facial emotions can be recognized using convolutional neural networks, snake and flask.
Facial expressions are the vital identifiers for human feelings, because it corresponds to the emotions. 
Most of the times (roughly in 55% cases), the facial expression is a nonverbal way of emotional expression, and it can be considered as concrete evidence to uncover whether an individual is speaking the truth or not.

## Mission objectives

- Be able to analyze and classify images of faces according to the facial-expression.
- Be able to analyze real-time images (video streaming) and implement facial-expression recognition.
- Explore techniques to identify emotions from subtle movements or gestures on the face.


## Coding Structure:

- Import the required Packages and Libraries.
- Data analysis and Creating Training and Validation Batches.
- Create a CNN using 4 Convolutional Layers including *Batch Normalization*,
*Activation*, *Max Pooling*, *Dropout* Layers followed by *Flatten* Layer, 2 Fully
*Connected dense* Layers and finally Dense Layer with *SoftMax* Activation
Function.
- Compile the model using `Adam` Optimizer and categorical cross entropy
loss function.
- Training the model for 15 epochs and then Evaluating the model as well as
saving the model Weights in `.h5` Values
- Saving the model as `JSON` string.
- Creating a Class in a separate file to reload the model and its weights to
make predictions and return the probabilities of each emotion.
- Creating one more class in a Separate file which takes in the `Real-time
Video input` and returns frames of Images with a Circle detecting the face
and putting text of its emotion on it.
- A python script is also created which upon running yields the `Graphical`
`Visualization` of Emotions present in the Image provided.
- Finally creating a file which inherits form all the Classes defined by us and
deploys our application using *Flask*.
<img src="[https://miro.medium.com/max/1864/1*oURfHMP1--ttXnDx0heusg.png](https://www.mdpi.com/sensors/sensors-20-02393/article_deploy/html/images/sensors-20-02393-g003.png)">



## The Mission

An important global firm receives thousands of job applications every year. However, the HR team does not have
enough time to review each one of the applications. This is the reason why they are looking for innovative solutions
to be integrated into the selection and recruitment process.

Looking to accelerate the interview pace, the company is investing resources in a video-interview system where pre-defined questions are asked by a virtual HR agent. In some jobs, personality is an important asset and the company would like to automatically analyze the images from the video footage, quantify the emotions expressed by the job applicants, and select the appropriate candidate according to the open job opportunities. They want to take it a step further, detecting a smile or sad face is not enough. They want a tool capable of recognizing even subtle changes in facial expression that might
indicate a particular emotion.

![Emotions (GIF)](https://media.giphy.com/media/84rG9j2H62hwc/giphy.gif)


### Must-have features

- As a minimum valuable product, the model should be able to identify human expressions such as happiness and sadness.
- Explore ways to detect more complex emotions.
- Deployment of the tool or integration with platforms is highly encouraged.

### Miscellaneous information

Some datasets are provided as initial material helpful to train or test your models. However, take the time to think
about the limitations that might be attached to the provided data and explore the possibility of using more datasets
or technologies adapted to the problem you are trying to solve.

### Dataset

- [FER-2013](https://www.kaggle.com/msambare/fer2013)


## Technical Evaluation criteria

- Ability to recognize sadness and happiness in images.
- Ability to recognize more complex emotions in images.
- Ability to recognize emotions in videos.
- A baseline model was established.
- Appropiate metrics were used to evaluate the model.
- Preprocessing of the images was done to improve detection.
- Model was deployed using Streamlit, Flask, and/or Heroku.


