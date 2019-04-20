from nn.convFullNN import ConvFullNN
from nn.convFastNN import ConvFastNN
import cv2
import os
import numpy
from PIL import Image
from shutil import copyfile
from utils import pfm
import data

from capi import pyelas

def convFulNN_compute(results_fname, sample_name, w_filename, max_disp):
    dnet = ConvFullNN("convFullNN", results_fname)
    dnet.max_disp = max_disp

    dnet.createResultDir(sample_name)
    left_pic = Image.open(results_fname + sample_name + "/im0.png")
    ctc_width, ctc_height = left_pic.size

    dnet.createCTCModels(ctc_height, ctc_width)
    dnet.loadCTCWeights(w_filename)
    dnet.convolve_images_ctc(results_fname + sample_name)

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

def convFastNN_compute(results_fname, sample_name, w_filename, max_disp):
    dnet = ConvFastNN("convFastNN", results_fname)
    dnet.max_disp = max_disp

    dnet.createResultDir(sample_name)
    left_pic = Image.open(results_fname + sample_name + "/im0.png")
    ctc_width, ctc_height = left_pic.size

    dnet.createCTCModels(ctc_height, ctc_width)
    dnet.loadCTCWeights(w_filename)
    dnet.convolve_images_ctc(results_fname + sample_name)

    # get predictions
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

def pyelas_compute(results_fname, sample_name, max_disp):
    
    image_left_filename  = results_fname + sample_name + "/im0"
    image_right_filename = results_fname + sample_name + "/im1"

    im = Image.open(image_left_filename + ".png")
    im = im.convert("L")
    im.save(image_left_filename + ".pgm")

    im = Image.open(image_right_filename + ".png")
    im = im.convert("L")
    im.save(image_right_filename + ".pgm")

    pyelas.process(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm", max_disp, 0)

    handler_name = "ELAS"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    outDispFileName      = sgbm_results_fname + "calc_disp.png"

    os.makedirs(sgbm_results_fname, exist_ok=True) 

    copyfile("disp.pfm", sgbm_results_fname + "calc_disp.pfm")
    os.remove("disp.pfm")

    disp_pfm = pfm.readPfm(sgbm_results_fname + "calc_disp.pfm")
    cv2.imwrite(outDispFileName, disp_pfm)

    cv2.imwrite(sgbm_results_fname + "calc_disp_norm.png", 255*disp_pfm/max_disp)