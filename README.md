# TF_disp

This repository contains the research project on disparity map computation using deep learning.
Frameworks used are Keras + TensorFlow.

Folder structure:

* *samples* directory contains images to process. Left and right images must be named **im0.png** and **im1.png** respectively.
* *disp_nn* directory contains code and other directories. 
    - *img* directory contains processed images and disparity maps as well as information about model in *model*.
    - *np_data* directory contains convolved images as numpy arrays. It is zipped as it is very large.
    - *weights* directory contains weights for the networks.

Training is done by **acc1_teach.py**. Several parameters of the network can be adjusted.

Testing is done by **acc1_test_part_gpu.py**. Maximum disparity and image name must be set before testing.

## References

Our work was inspired by this papers:

1. A. Geiger, M. Roser and R. Urtasun. Efficient Large-Scale Stereo Matching. Asian Conference on Computer Vision (ACCV), 2010.
2. Jure Žbontar and Yann LeCun. Stereo matching by training a convolutional neural network to compare image patches. J. Mach. Learn. Res. 17, 1 (January 2016), 2287-2318.


