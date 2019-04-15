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

class ConvFullNN:

    lctc_model = None
    rctc_model = None
    dtc_model  = None

    conv_left_patches  = None
    conv_right_patches = None
    
    conv_feature_maps = 112
    dense_size = 384
    patch_size = 11
    max_disp   = 60
    w_filename = ""
    name       = ""
    results_fname = ""

    def __init__(self, name, results_fname):
        self.name = name
        self.results_fname = results_fname

    def createResultDir(self, sample_name):
        os.makedirs(self.results_fname + sample_name + "/" + self.name + "/", exist_ok=True) 

    def createCTCModels(self, ctc_height, ctc_width):

        conv_feature_maps = self.conv_feature_maps

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

        self.lctc_model = Model(inputs=ctc_left_input, outputs=ctc_left_flatten)
        self.rctc_model = Model(inputs=ctc_right_input, outputs=ctc_right_flatten)

    def loadCTCWeights(self, filename):
        self.w_filename = filename
        self.lctc_model.load_weights(filename, by_name = True)
        self.rctc_model.load_weights(filename, by_name = True)

    def createDTCModel(self):
        patch_size = self.patch_size
        conv_feature_maps = self.conv_feature_maps
        dense_size = self.dense_size

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
        model.load_weights(self.w_filename, by_name = True)

        # replacing dense model with convolutional (dtc: dense to convolutional)
        dtc_height = self.conv_left_patches.shape[0]
        dtc_width  = self.conv_left_patches.shape[1] - self.max_disp
        dtc_input  = Input(shape=(dtc_height, dtc_width, conv_feature_maps*2))
        w,b = model.get_layer("d2").get_weights()
        new_w = numpy.expand_dims(numpy.expand_dims(w, axis = 0), axis = 0)
        dtc_layer = Conv2D(dense_size, kernel_size=1, activation="relu", name="dtc1", weights=[new_w,b])(dtc_input)
        w,b = model.get_layer("d3").get_weights()
        new_w = numpy.expand_dims(numpy.expand_dims(w, axis = 0), axis = 0)
        dtc_layer = Conv2D(dense_size, kernel_size=1, activation="relu", name="dtc2", weights=[new_w,b])(dtc_layer)
        w,b = model.get_layer("d4").get_weights()
        new_w = numpy.expand_dims(numpy.expand_dims(w, axis = 0), axis = 0)
        dtc_output = Conv2D(1, kernel_size=1, activation="sigmoid", name="dtc_out", weights=[new_w,b])(dtc_layer)
        self.dtc_model = Model(inputs = dtc_input, outputs = dtc_output)

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
        dtc_predictions = self.dtc_model.predict([numpy.expand_dims(numpy.concatenate((self.conv_left_patches[::, max_disp:width,::],
                                                                                self.conv_right_patches[::,0:width-max_disp,::]), 
                                                                                axis = 2), axis = 0)])
        dtc_predictions = numpy.squeeze(dtc_predictions, axis=0)
        for i in range(1, max_disp):
            prediction = self.dtc_model.predict([numpy.expand_dims(numpy.concatenate((self.conv_left_patches[::, max_disp:width,::],
                                                                                self.conv_right_patches[::,i:width-max_disp+i,::]), 
                                                                                axis = 2), axis = 0)])
            dtc_predictions = numpy.concatenate((dtc_predictions, numpy.squeeze(prediction, axis=0)), axis=2)
            print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % (100*(i+1)/max_disp), "%", end = "\r")

        print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % 100, "%", end = "\r")
        numpy.save(predictions_filename, dtc_predictions)
        print("\ntotal time ", "%.2f" % (time.time()-timestamp))
        disp_pix[::, max_disp:width] = (255*(max_disp - numpy.argmax(dtc_predictions, axis = 2)))/max_disp
        img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
        img.save(results_fname + "/" + sample_name + "_raw_disp.png", "PNG")

    def getReportData(self):
        pass
