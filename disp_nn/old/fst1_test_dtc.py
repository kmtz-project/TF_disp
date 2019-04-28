import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
import numpy
from PIL import Image, ImageDraw
import time
import data
import utils
import cv2

# constants
conv_feature_maps = 112
dense_size = 384
patch_size = 9
max_disp = 60
image_name = "Motorcycle"
match_th = 0.2
error_threshold = 12
cpu_only = True

# fix random seed for reproducibility
numpy.random.seed(7)

# set GPU use
# if cpu_only:
#     utils.set_gpu(False)

# load convolved images
left = numpy.load('np_data/' + image_name + '_left_conv.npy')
right = numpy.load('np_data/' + image_name + '_right_conv.npy')

# compute disparity map
# data.disp_map_from_conv_fst(left, right, patch_size, max_disp, match_th, conv_feature_maps, image_name + "_disp_fst")

net_results_fname = "./np_data/"
predictions_filename = net_results_fname + image_name + '_disp_fst_predictions.npy'

print(predictions_filename)
predictions = numpy.load(predictions_filename)

img = data.sgbm(predictions, max_disp, 30)
cv2.imwrite("./work/" + image_name + "_fst_calc_disp.png", img)

disp_pix = numpy.zeros((predictions.shape[0], predictions.shape[1]+max_disp), dtype=numpy.float)
disp_pix[:, max_disp:] = (255*(max_disp - numpy.argmin(1-predictions, axis = 2)))/max_disp

cv2.imwrite("./work/" + image_name + "_fst_calc_disp2.png", disp_pix)

##############################################################################################
#left, right = data.get_random_sample("../samples/cones/", 11, 8, 12, 4, 0, 5, 68)
