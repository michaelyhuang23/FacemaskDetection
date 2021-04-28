## Description

This is a facemask detection model using YOLO frontend and Fast RCNN backend. 

We use YOLO's pixel-wise objectness prediction as the region proposer for Fast RCNN. Pixels of different sizes (via using different-sized filters) correspond to different sized RoI at those locations. This architecture is tailored towards face detection as it works best with "squarely" objects.

Locations with a objectness>0.5 are fed into the Fast RCNN backend for bounding box regression and image classification (as properly weared facemask, improperly weared facemask, no facemask)

The model is built-upon InceptionV3 pretrained on ImageNet



The Dataset is taken from Kaggle

