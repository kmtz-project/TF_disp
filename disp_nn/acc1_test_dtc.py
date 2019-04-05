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

# constants
conv_feature_maps = 112
dense_size = 384
patch_size = 11
max_disp = 60
image_name = "pattern1"
match_th = 0.2
error_threshold = 12
cpu_only = True

# fix random seed for reproducibility
numpy.random.seed(7)

# set GPU use
if cpu_only:
    utils.set_gpu(False)

# create full model
left_input = Input(shape=(patch_size, patch_size, 1, ))
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc1") (left_input)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc2") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc3") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc4") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc5") (left_conv)
left_flatten = Flatten(name = "left_flatten_layer")(left_conv)

right_input = Input(shape=(patch_size, patch_size, 1, ))
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc1") (right_input)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc2") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc3") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc4") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc5") (right_conv)
right_flatten = Flatten(name = "right_flatten_layer")(right_conv)

conc_layer = Concatenate(name = "d1")([left_flatten, right_flatten])
dense_layer = Dense(dense_size, activation="relu", name = "d2")(conc_layer)
dense_layer = Dense(dense_size, activation="relu", name = "d3")(dense_layer)
output_layer = Dense(1, activation="sigmoid", name = "d4")(dense_layer)

model = Model(inputs=[left_input, right_input], outputs=output_layer)
model.load_weights("weights/acc1_weights6.h5", by_name = True)

# load convolved images
left = numpy.load('np_data/' + image_name + '_left_conv.npy')
right = numpy.load('np_data/' + image_name + '_right_conv.npy')

# replacing dense model with convolutional (dtc: dense to convolutional)
dtc_height = left.shape[0]
dtc_width = left.shape[1] - max_disp
dtc_input = Input(shape=(dtc_height, dtc_width, conv_feature_maps*2))
w,b = model.get_layer("d2").get_weights()
new_w = numpy.expand_dims(numpy.expand_dims(w, axis = 0), axis = 0)
dtc_layer = Conv2D(dense_size, kernel_size=1, activation="relu", name="dtc1", weights=[new_w,b])(dtc_input)
w,b = model.get_layer("d3").get_weights()
new_w = numpy.expand_dims(numpy.expand_dims(w, axis = 0), axis = 0)
dtc_layer = Conv2D(dense_size, kernel_size=1, activation="relu", name="dtc2", weights=[new_w,b])(dtc_layer)
w,b = model.get_layer("d4").get_weights()
new_w = numpy.expand_dims(numpy.expand_dims(w, axis = 0), axis = 0)
dtc_output = Conv2D(1, kernel_size=1, activation="sigmoid", name="dtc_out", weights=[new_w,b])(dtc_layer)
dtc_model = Model(inputs = dtc_input, outputs = dtc_output)

# compute disparity map
data.disp_map_from_conv_dtc(left, right, patch_size, max_disp, match_th, conv_feature_maps, dtc_model, image_name + "_disp_dtc")

##############################################################################################
#left, right = data.get_random_sample("../samples/cones/", 11, 8, 12, 4, 0, 5, 68)
