# TF_disp
This repository contains the research project on disparity map computation using deep learning.\
Frameworks used are Keras + TensorFlow.\
*samples* directory contains images to process. It is necessary for left image to be named **im0.png** \
and for right image - **im1.png**.\
*disp_nn* directory contains code and other directories. \
*img* directory contains processed images and disparity maps as well as information about model in *model*.\
*np_data* directory contains convolved images as numpy arrays. It is zipped as it is very large.\
*weights* directory contains weights for the networks.\
Training is done by **acc1_teach.py**. Several parameters of the network can be adjusted.\
Testing is done by **acc1_test_part_gpu.py**. Maximum disparity and image name must be set before testing.
