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

class ConvFastNN:

    lctc_model = None
    rctc_model = None
    dtc_model  = None

    conv_left_patches  = None
    conv_right_patches = None
    
    conv_feature_maps = 112
    dense_size = 384
    patch_size = 9
    max_disp   = 60
    w_filename = ""
    name       = ""
    results_fname = ""

    def __init__(self, name, results_fname):
        self.name = name
        self.results_fname = results_fname

    def createResultDir(self, sample_name):
        os.makedirs(self.results_fname + sample_name + "/" + self.name + "/", exist_ok=True) 

    def loadCTCWeights(self, filename):
        self.w_filename = filename
        self.lctc_model.load_weights(filename, by_name = True)
        self.rctc_model.load_weights(filename, by_name = True)

    def createCTCModels(self, ctc_height, ctc_width):

        conv_feature_maps = self.conv_feature_maps

        ctc_left_input = Input(shape=(ctc_height, ctc_width, 1, ))
        ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc1") (ctc_left_input)
        ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc2") (ctc_left_conv)
        ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc3") (ctc_left_conv)
        ctc_left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc4") (ctc_left_conv)
        ctc_left_flatten = Flatten(name = "lf")(ctc_left_conv)

        ctc_right_input = Input(shape=(ctc_height, ctc_width, 1, ))
        ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc1") (ctc_right_input)
        ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc2") (ctc_right_conv)
        ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc3") (ctc_right_conv)
        ctc_right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc4") (ctc_right_conv)
        ctc_right_flatten = Flatten(name = "rf")(ctc_right_conv)

        self.lctc_model = Model(inputs=ctc_left_input, outputs=ctc_left_flatten)
        self.rctc_model = Model(inputs=ctc_right_input, outputs=ctc_right_flatten)

    def convolve_images_ctc(self, folder_name):

        patch_size = self.patch_size
        conv_feature_maps = self.conv_feature_maps

        print("begin convolution")
        left_pic  = Image.open(folder_name + "/im0.png").convert("L")
        right_pic = Image.open(folder_name + "/im1.png").convert("L")
        border = int(patch_size/2)
        
        left_pix = numpy.atleast_3d(left_pic)
        right_pix = numpy.atleast_3d(right_pic)
        width, height = left_pic.size
        self.conv_left_patches = numpy.zeros((height,width,conv_feature_maps))
        self.conv_right_patches = numpy.zeros((height,width,conv_feature_maps))
        
        left_f = lambda x: (x - left_pix.mean())/left_pix.std()
        norm_left = left_f(left_pix)
        right_f = lambda x: (x - right_pix.mean())/right_pix.std()
        norm_right = right_f(right_pix)
        
        timestamp = time.time()
        l_prediction = self.lctc_model.predict([[norm_left]])
        r_prediction = self.rctc_model.predict([[norm_right]])
        self.conv_left_patches[border:-border,border:-border,::] = l_prediction.reshape((height - 2*border, width - 2*border, conv_feature_maps))
        self.conv_right_patches[border:-border,border:-border,::] = r_prediction.reshape((height - 2*border, width - 2*border, conv_feature_maps))
        print("total time ", round(time.time()-timestamp, 3))


    def disp_map_from_conv_dtc(self, sample_name):
        print("begin disparity computation")
        results_fname = self.results_fname + "/" + sample_name + "/" + self.name + "/"
        predictions_filename = results_fname + sample_name + '_predictions'
        max_disp = self.max_disp
        height = self.conv_left_patches.shape[0]
        width = self.conv_right_patches.shape[1]

        disp_pix = numpy.zeros((height,width))
        timestamp = time.time()
        cosine_arr =  numpy.einsum('ijk,ijk->ij', self.conv_left_patches[::, max_disp:width,::], self.conv_right_patches[::,0:width-max_disp,::]) / (
                numpy.linalg.norm(self.conv_left_patches[::, max_disp:width,::], axis=-1) * numpy.linalg.norm(self.conv_right_patches[::,0:width-max_disp,::], axis=-1))
        cosine_arr = numpy.expand_dims(cosine_arr, axis=2)
        print(cosine_arr.shape)
        for i in range(1, max_disp):
            cosine =  numpy.einsum('ijk,ijk->ij', self.conv_left_patches[::, max_disp:width,::], self.conv_right_patches[::,i:width-max_disp+i,::]) / (
                numpy.linalg.norm(self.conv_left_patches[::, max_disp:width,::], axis=-1) * numpy.linalg.norm(self.conv_right_patches[::,i:width-max_disp+i,::], axis=-1))
            cosine = numpy.expand_dims(cosine, axis=2)
            cosine_arr = numpy.concatenate((cosine_arr, cosine), axis=2)
            print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % (100*(i+1)/max_disp), "%", end = "\r")
        cosine_arr[numpy.isnan(cosine_arr)] = 0
        print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % 100, "%", end = "\r")
        numpy.save(predictions_filename, cosine_arr)
        disp_pix[::, max_disp:width] = (255*(max_disp - numpy.argmax(cosine_arr, axis = 2)))/max_disp
        print("\ntotal time ", "%.2f" % (time.time()-timestamp))

        img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
        img.save(results_fname + "/" + sample_name + "_raw_disp.png", "PNG")