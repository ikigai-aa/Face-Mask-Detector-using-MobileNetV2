# Face Mask Detector using MobileNetV2 :india:

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Resources](#resources)
  
  
## Demo
Link: [https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/demo%20video.mp4](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/demo%20video.mp4)


## Overview
This is a simple image classification project trained on the top of Keras/Tensorflow API with MobileNetV2 deep neural network architecture having weights considered as pre-trained 'imagenet' weights. The trained model (`mask-detector-model.h5`) takes the real-time video from webcam as an input and predicts if the face landmarks in Region of Interest (ROI) is 'Mask' or 'No Mask' with real-time on screen accuracy.


## Motivation

Globally, the coronavirus stats says it has more than 16.3M confirmed cases and claimed over 649K lives so far, according to the Johns Hopkins University when I am writing this project repo. As many as 9.41 M people have recovered.

India’s coronavirus cases are increasing in an unimaginable rate and is breaking the record in the highest single-day increase so far every new day.. The country’s tally rose to 14,35,453 and the toll stood at 32,771. India is now the third worst-affected country by the pandemic and has overtaken Italy, according to Johns Hopkins University.

The World Health Organization said that new information showed that protective masks could be a barrier for potentially infectious droplets. The coronavirus primarily spreads through the transmission of respiratory droplets from infected people.On 5th day of June changed its guidelines about the use of protective face masks in public, saying that they must be worn at all places where physical distancing is not possible. The global health body had said in April that there was not enough evidence to show that healthy people should wear masks to shield themselves from the coronavirus.

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/images/WHO.png)


Link: [https://twitter.com/i/status/1268986094042992640](https://twitter.com/i/status/1268986094042992640)


WHO also said that high-risk groups should wear medical grade masks in cases where physical distancing is not possible.Several countries, including India, have made wearing masks in public compulsory. In many states, people have been fined for not wearing masks. Maintaining hygiene and using protective equipment has become even more important ahead of the reopening of religious places, malls and restaurants in India from next week.

This motivated me to create a the COVID-19 Mask Detector with some of my ML/DL skills and making it such accurate that it could potentially be used to help ensure your safety and the safety of others (Leaving it on to the medical professionals to decide on, implement in public places).


## Technical Aspect
In order to train a Face Mask Detector, we need to break our project into two distinct phases, each with its own respective sub-steps:

Training: Here we’ll focus on loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the face mask detector to disk

Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with_mask or without_mask

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/images/face_mask_detection_phases.png)\


### Dataset Resource:

Link: https://drive.google.com/drive/folders/1FHPJRCab-cyLq8IVz83LkU71gOc7gTS8?usp=sharing

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/images/face_mask_detection_dataset.jpg)


### Project structure

```
├── dataset
│   ├── with_mask [690 entries]
│   └── without_mask [686 entries]
├── examples
│   ├── example_01.png
│   ├── example_02.png
│   └── example_03.png
├── face_detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── detect_mask_image.py
├── detect_mask_video.py
├── mask_detector.model
├── evaluation.png
└── Data Augmentation and Model Training.ipynb
├── requirements.txt
└── mask-detector-model.model
```


### Important Python Scripts:

1. Data Augmentation and Preprocessing.ipynb: In this notebook Accepts our dataset is taken as input and fine-tuning is donw with MobileNetV2 DNN architecture upon it to create our mask-detector-model.model. A training history evaluation.png containing accuracy/loss curves is also produced for better visualization of Model Evaluation through a plot.Some important processes which we performed here:

a. Data augmentation
b. Loading the MobilNetV2 classifier (we will fine-tune this model with pre-trained ImageNet weights)
c. Building a new fully-connected (FC) head
d. Pre-processing
e. Loading image data

Libraries Significance:

scikit-learn: for binarizing class labels, segmenting our dataset, and printing a classification report.
imutils: To find and list images in our dataset. 
matplotlib: To plot our training curves.

2. detect_mask_from_webcam.py: Using your webcam, this script applies face mask detection to every frame in the stream using webcom to read the real-time video.

Some command line arguments in this script include:
```
--image: The path to the input image containing faces for inference
--face: The path to the face detector model directory (we need to localize faces prior to classifying them)
--model: The path to the face mask detector model that we trained earlier in this tutorial
--confidence: An optional probability threshold can be set to override 50% to filter weak face detections
```


## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

```bash
## Run
> STEP 1
After unzipping the forked zip file of this project into your local machine, type the follwing command from the directory where you saved the project files in the command prompt: 
pip install -r requirements.txt

> STEP 2
Open Jupyter Notebook and run Data Augmentation and Preprocessing.ipynb in order to train your custom dataset within your loacl machine and preprocess the images meanwhile.

> STEP 3
Run detect_mask_from_webcam.py from the same directory of your project folder in the command prompt in order to test the detector in real- time using the webcam.
```

## Results/Classification Report

```
              precision    recall  f1-score   support

   with_mask       0.97      1.00      0.99       138
without_mask       1.00      0.97      0.99       138

    accuracy                           0.99       276
   macro avg       0.99      0.99      0.99       276
weighted avg       0.99      0.99      0.99       276

```

## Accuracy/Loss Plot

![](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/blob/master/evaluation.png)\

...
## To Do
1. This approach reduces our computer vision pipeline to a single step — rather than applying face detection and then our face mask detector model, all we need to do is apply the object detector to give us bounding boxes for people both with_mask and without_mask in a single forward pass of the network.

2. An integration of this project to a web app/android app.


## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/ikigai-aa/Face-Mask-Detector-using-MobileNetV2/issues/new). Please include sample queries and their corresponding results.


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/) 

[<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=170>](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

[<img target="_blank" src="https://www.gstatic.com/devrel-devsite/prod/vbf66214f2f7feed2e5d8db155bab9ace53c57c494418a1473b23972413e0f3ac/tensorflow/images/lockup.svg" width=280>](https://www.gstatic.com/devrel-devsite/prod/vbf66214f2f7feed2e5d8db155bab9ace53c57c494418a1473b23972413e0f3ac/tensorflow/images/lockup.svg)

[<img target="_blank" src="http://image-net.org/index_files/logo.jpg" width=200>](http://image-net.org/index_files/logo.jpg) 

[<img target="_blank" src="https://jupyter.org/assets/nav_logo.svg" width=200>](https://jupyter.org/assets/nav_logo.svg) 


## Team
Ashish Agarwal

LinkedIn Profile: [https://www.linkedin.com/in/ashish-agarwal-502203113/](https://www.linkedin.com/in/ashish-agarwal-502203113/)

## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2020 Ashish Agarwal

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Resources

1. https://www.who.int/publications/i/item/advice-on-the-use-of-masks-in-the-community-during-home-care-and-in-healthcare-settings-in-the-context-of-the-novel-coronavirus-(2019-ncov)-outbreak
2. https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
3. http://www.image-net.org/
4. https://arxiv.org/abs/1801.04381
