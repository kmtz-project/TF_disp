import os
from nn.convFullNN import ConvFullNN
from PIL import Image
import data
import numpy
from shutil import copyfile

def convFulNN_compute(results_fname, sample_name, w_filename):
    dnet = ConvFullNN("convFullNN", results_fname)

    dnet.createResultDir(sample_name)
    left_pic = Image.open(results_fname + sample_name + "/im0.png")
    ctc_width, ctc_height = left_pic.size

    dnet.createCTCModels(ctc_height, ctc_width)
    dnet.loadCTCWeights(w_filename)
    dnet.convolve_images_ctc(sample_fname)

    # get predictions
    dnet.createDTCModel()
    dnet.disp_map_from_conv_dtc(sample_name)

    # get disp from predictions
    net_results_fname = results_fname + sample_name + "/" + dnet.name + "/"
    predictions_filename = net_results_fname + sample_name + '_predictions.npy'
    predictions = numpy.load(predictions_filename)
    img = data.sgbm(predictions, dnet.max_disp, 10)
    img.save(net_results_fname + sample_name + "_sgbm.png", "PNG")

    del dnet

# set log level ('0' - all messages, '1' - no info messages, '2' - no warning messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

samples_fname  = "../samples/Middlebury_scenes_2014/trainingQ/"
sample_name    = "Teddy"
sample_fname   = samples_fname + sample_name
w_filename     = "weights/acc1_weights6.h5"

# Create results folder
results_fname = "../results/"
os.makedirs(results_fname + sample_name, exist_ok=True) 
copyfile(sample_fname + "/im0.png", results_fname + sample_name + "/im0.png")
copyfile(sample_fname + "/im1.png", results_fname + sample_name + "/im1.png")
#----

convFulNN_compute(results_fname, sample_name, w_filename)

