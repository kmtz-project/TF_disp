import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
import numpy
from PIL import Image, ImageDraw
import time
import os
import data
import utils

# set log level ('0' - all messages, '1' - no info messages, '2' - no warning messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# constants
conv_feature_maps = 112
dense_size = 384
patch_size = 11
image_name = "kron"
cpu_only = True

# fix random seed for reproducibility
numpy.random.seed(7)

# set GPU use
if cpu_only:
    utils.set_gpu(False)

# create convolutional part models
left_pic = Image.open("../samples/" + image_name + "/im0.png")
ctc_width, ctc_height = left_pic.size

ctc_left_input = Input(shape=(ctc_height, ctc_width, 1, ))
ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc1") (ctc_left_input)
ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc2") (ctc_left_conv)
ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc3") (ctc_left_conv)
ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc4") (ctc_left_conv)
ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc5") (ctc_left_conv)
ctc_left_flatten = Flatten(name = "lf")(ctc_left_conv)

ctc_right_input = Input(shape=(ctc_height, ctc_width, 1, ))
ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc1") (ctc_right_input)
ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc2") (ctc_right_conv)
ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc3") (ctc_right_conv)
ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc4") (ctc_right_conv)
ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc5") (ctc_right_conv)
ctc_right_flatten = Flatten(name = "rf")(ctc_right_conv)

lctc_model = Model(inputs=ctc_left_input, outputs=ctc_left_flatten)
rctc_model = Model(inputs=ctc_right_input, outputs=ctc_right_flatten)
lctc_model.load_weights("weights/acc1_weights6.h5", by_name = True)
rctc_model.load_weights("weights/acc1_weights6.h5", by_name = True)

# convolve images
left, right = data.convolve_images_ctc("../samples/" + image_name + "/", patch_size, conv_feature_maps, lctc_model, rctc_model)

# save convolved images
numpy.save('np_data/' + image_name + '_left_conv', left)
numpy.save('np_data/' + image_name + '_right_conv', right)
