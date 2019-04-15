import os
from nn.convFullNN import ConvFullNN
from PIL import Image
import data
import numpy
from shutil import copyfile
import cv2
import colorama
from termcolor import colored

def convFulNN_compute(results_fname, sample_name, w_filename, max_disp):
    dnet = ConvFullNN("convFullNN", results_fname)
    dnet.max_disp = max_disp

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
    img = data.sgbm(predictions, dnet.max_disp, 30)
    cv2.imwrite(net_results_fname + "calc_disp.png", img)

    del dnet

def opencvSGBM_compute(results_fname, sample_name, max_disp):
    window_size = 3
    min_disp = 0
    stereo = cv2.StereoSGBM_create(
        minDisparity = 0,
        numDisparities = max_disp,
        blockSize = 2,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 2,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    handler_name = "opencvSGBM"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    image_left_filename  = results_fname + sample_name + "/im0.png"
    image_right_filename = results_fname + sample_name + "/im1.png"
    outDispFileName      = sgbm_results_fname + "calc_disp.png"

    os.makedirs(sgbm_results_fname, exist_ok=True) 

    image_left = cv2.imread(image_left_filename)
    image_right = cv2.imread(image_right_filename)

    # compute disparity
    disparity = stereo.compute(image_left, image_right).astype(numpy.float32) / 16.0

    print(disparity.max())

    #disparity_out = (disparity)/num_disp * 256
    cv2.imwrite(outDispFileName, disparity)

# set log level ('0' - all messages, '1' - no info messages, '2' - no warning messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
colorama.init()

samples_list   = ["Adirondack", "ArtL", "Motorcycle", "Piano", "Recycle", "Shelves", "Teddy"]
samples_fname  = "../samples/Middlebury_scenes_2014/trainingQ/"
w_filename     = "weights/acc1_weights6.h5"
max_disp       = 64
results_fname  = "../results/"

for sample_name in samples_list:

    print(colored("-> " + sample_name, 'green'))
    print("-"*50)
    sample_fname   = samples_fname + sample_name
    
    # Create results folder
    os.makedirs(results_fname + sample_name, exist_ok=True) 
    copyfile(sample_fname + "/im0.png", results_fname + sample_name + "/im0.png")
    copyfile(sample_fname + "/im1.png", results_fname + sample_name + "/im1.png")
    #----

    print("handler_name: "+ colored("convFulNN", 'yellow'))
    convFulNN_compute(results_fname, sample_name, w_filename, max_disp)

    print("handler_name: "+ colored("opencvSGBM", 'yellow'))
    opencvSGBM_compute(results_fname, sample_name, max_disp)

