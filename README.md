# myEnhancer

## What it is
Script, that performes an resolution enhancement on images using pre-trained machine learning models.

## Mashine learning models
Pre-trained mashine learning models were provided by [Saafke](https://github.com/Saafke) via his GitHub. Link to his repository is available [here](https://github.com/Saafke/EDSR_Tensorflow/tree/master).

## Original project
This project is a modification of a group project created for a subject called "Technika Obrazowa" ("Imaging Technique") at Warsaw University of Technology.

## Requirements
This program raquires OpenCV library installation. I highly recommend to do it via pip installation with a command prompt `pip3 install opencv-python`

## Limitations (one day I'll fix them!)
While using this program, you have to have in mind that it has limitations. EDSR model gives (in my opinion) the best resoults, but it takes much more time to finish, than other models. Also, it has limitations in original image sizes - it work only with small images. For bigger Images I reccomend using different models.
