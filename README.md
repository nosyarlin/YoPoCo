# CV-pose-detection
This project is concerned with verifying whether the user is performing yoga poses correctly. To do so, we extract the user's pose information from an image of him using OpenPose and compare it with the target yoga pose. If the two poses have a high similarity score, we treat the user's pose as correct. 

# Files
We compared two methods of calculating similarity scores. They are namely using cosine similarity and using a neural network. 

+ ComparatorNet.h5: Contains our neural network
+ ComparatorNet.ipynb: Jupyter notebook used to build our network along with its evaluation
+ Cosine similarity.ipynb: Jupyter notebook used to evaluate the cosine similarity method
+ get_openpose_coords.py: Script to run through our entire image dataset and extract pose coordinates using OpenPose
+ model_history_log.csv: Allows me to plot loss and accuracy over time as I trained the neural network
+ demo.py: Demo file. It's pretty cool to run if you have OpenPose installed properly

The coords directory contains all the pose coordinates extracted from our image dataset. The image dataset will not be made available here. 

# Getting Started
In order to run the demo file, you will need to install OpenPose first. 
