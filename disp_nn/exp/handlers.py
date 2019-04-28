from nn.convFullNN import ConvFullNN
from nn.convFastNN import ConvFastNN
import cv2
import os
import numpy
from PIL import Image
from shutil import copyfile
from utils import pfm
import data

from capi import pyelas, elasCNN, elasCNNsup, elasCNNstd

def convFulNN_compute(results_fname, sample_name, w_filename, max_disp):
    handler_name = "convFullNN"
    dnet = ConvFullNN("convFullNN", results_fname)
    dnet.max_disp = max_disp

    dnet.createResultDir(sample_name)
    left_pic = Image.open(results_fname + sample_name + "/im0.png")
    ctc_width, ctc_height = left_pic.size

    dnet.createCTCModels(ctc_height, ctc_width)
    dnet.loadCTCWeights(w_filename)
    dnet.convolve_images_ctc(results_fname + sample_name)
    numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv",dnet.conv_left_patches)
    numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv",dnet.conv_right_patches)

    # get predictions
    dnet.createDTCModel()
    dnet.disp_map_from_conv_dtc(sample_name)

    # get disp from predictions
    net_results_fname = results_fname + sample_name + "/" + dnet.name + "/"
    predictions_filename = net_results_fname + sample_name + '_predictions.npy'
    predictions = numpy.load(predictions_filename)
    img = data.sgbm(predictions, dnet.max_disp, 30)
    cv2.imwrite(net_results_fname + "calc_disp.png", img)
    calc_disp=cv2.imread(net_results_fname + "calc_disp.png")
    calc_disp = calc_disp.astype("int32")
    calc_disp[:, max_disp:] = (255*calc_disp[:, max_disp:])/max_disp
    cv2.imwrite("../results/" + sample_name + "/" + handler_name + "/calc_disp_norm.png", calc_disp)

    del dnet

def convFastNN_compute(results_fname, sample_name, w_filename, max_disp):
    handler_name = "convFastNN"
    dnet = ConvFastNN("convFastNN", results_fname)
    dnet.max_disp = max_disp

    dnet.createResultDir(sample_name)
    left_pic = Image.open(results_fname + sample_name + "/im0.png")
    ctc_width, ctc_height = left_pic.size

    dnet.createCTCModels(ctc_height, ctc_width)
    dnet.loadCTCWeights(w_filename)
    dnet.convolve_images_ctc(results_fname + sample_name)
    numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv",dnet.conv_left_patches)
    numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv",dnet.conv_right_patches)

    # get predictions
    dnet.disp_map_from_conv_dtc(sample_name)

    # get disp from predictions
    net_results_fname = results_fname + sample_name + "/" + dnet.name + "/"
    predictions_filename = net_results_fname + sample_name + '_predictions.npy'
    predictions = numpy.load(predictions_filename)
    img = data.sgbm(predictions, dnet.max_disp, 30)
    cv2.imwrite(net_results_fname + "calc_disp.png", img)
    calc_disp=cv2.imread(net_results_fname + "calc_disp.png")
    calc_disp = calc_disp.astype("int32")
    calc_disp[:, max_disp:] = (255*calc_disp[:, max_disp:])/max_disp
    cv2.imwrite("../results/" + sample_name + "/" + handler_name + "/calc_disp_norm.png", calc_disp)

    del dnet

def opencvSGBM_compute(results_fname, sample_name, max_disp):
    window_size = 3
    min_disp = 0
    sgbm_max_disp = max_disp
    if max_disp%16 > 0:
        sgbm_max_disp = 16*(int(max_disp/16)+1)
    print("sgbm_max_disp", sgbm_max_disp)
    stereo = cv2.StereoSGBM_create(
        minDisparity = 0,
        numDisparities = sgbm_max_disp,
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
    calc_disp=cv2.imread(outDispFileName)
    calc_disp = calc_disp.astype("int32")
    #max_disp = 60
    calc_disp[:, max_disp:] = (255*calc_disp[:, max_disp:])/max_disp
    cv2.imwrite("../results/" + sample_name + "/" + handler_name + "/calc_disp_norm.png", calc_disp)

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

def elasCNN_compute(results_fname, sample_name, max_disp, cosine_weight):
    
    image_left_filename  = results_fname + sample_name + "/im0"
    image_right_filename = results_fname + sample_name + "/im1"

    im = Image.open(image_left_filename + ".png")
    im = im.convert("L")
    im.save(image_left_filename + ".pgm")

    im = Image.open(image_right_filename + ".png")
    im = im.convert("L")
    im.save(image_right_filename + ".pgm")

    conv_left = numpy.load("../results/" + sample_name + "/convFastNN/" + sample_name + "_left_conv.npy")
    conv_right = numpy.load("../results/" + sample_name + "/convFastNN/" + sample_name + "_right_conv.npy")

    conv_left = conv_left.astype(dtype=numpy.float32)
    conv_right = conv_right.astype(dtype=numpy.float32)

    elasCNN.compute(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm", max_disp, 0, cosine_weight, conv_left, conv_right)

    handler_name = "ELAS-CNN"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    outDispFileName      = sgbm_results_fname + "calc_disp.png"

    os.makedirs(sgbm_results_fname, exist_ok=True) 

    copyfile("disp.pfm", sgbm_results_fname + "calc_disp.pfm")
    os.remove("disp.pfm")

    disp_pfm = pfm.readPfm(sgbm_results_fname + "calc_disp.pfm")
    cv2.imwrite(outDispFileName, disp_pfm)

    cv2.imwrite(sgbm_results_fname + "calc_disp_norm.png", 255*disp_pfm/max_disp)

def elasCNNsup_compute(results_fname, sample_name, max_disp, cosine_weight, support_cosine_weight, support_threshold, std_filter):
    
    image_left_filename  = results_fname + sample_name + "/im0"
    image_right_filename = results_fname + sample_name + "/im1"

    im = Image.open(image_left_filename + ".png")
    im = im.convert("L")
    im.save(image_left_filename + ".pgm")

    im = Image.open(image_right_filename + ".png")
    im = im.convert("L")
    im.save(image_right_filename + ".pgm")

    conv_left = numpy.load("../results/" + sample_name + "/convFastNN/" + sample_name + "_left_conv.npy")
    conv_right = numpy.load("../results/" + sample_name + "/convFastNN/" + sample_name + "_right_conv.npy")
    #mask = numpy.load("np_data/fst/" + sample_name + "_mask.npy")

    conv_left = conv_left.astype(dtype=numpy.float32)
    conv_right = conv_right.astype(dtype=numpy.float32)

    # to run this you need to replace elasCNNsup.pyd with elasCNNsup-wf.pyd (rename)
    #elasCNNsup.compute(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm",
                       #max_disp, 0, support_threshold, cosine_weight, std_filter, support_cosine_weight, mask, conv_left, conv_right)

    elasCNNsup.compute(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm",
                       max_disp, 0, support_threshold, cosine_weight, support_cosine_weight, conv_left, conv_right)

    handler_name = "ELAS-CNN-sup"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    outDispFileName      = sgbm_results_fname + "calc_disp.png"

    os.makedirs(sgbm_results_fname, exist_ok=True) 

    copyfile("disp.pfm", sgbm_results_fname + "calc_disp.pfm")
    os.remove("disp.pfm")

    disp_pfm = pfm.readPfm(sgbm_results_fname + "calc_disp.pfm")
    cv2.imwrite(outDispFileName, disp_pfm)

    cv2.imwrite(sgbm_results_fname + "calc_disp_norm.png", 255*disp_pfm/max_disp)

def elasCNNfusion_compute(results_fname, sample_name, max_disp, cosine_weight, window_size, std_th):
    
    elas_disp = Image.open(results_fname + sample_name + "/ELAS/calc_disp_norm.png").convert("L")
    raw_disp = Image.open(results_fname + sample_name + "/ConvFastNN/" + sample_name + "_raw_disp.png").convert("L")
    width, height = raw_disp.size
    raw_disp_pix = numpy.atleast_1d(raw_disp).astype('int')
    elas_disp_pix = numpy.atleast_1d(elas_disp).astype('int')
    std_pix = numpy.zeros((height, width))
    
    for i in range(int(window_size/2), height - int(window_size/2)):
        for j in range(int(window_size/2), width - int(window_size/2)):
            area = int((window_size-1)/2)
            std = raw_disp_pix[i-area:i+area+1,j-area:j+area+1].std()
            if std > std_th:
                std_pix[i,j] = elas_disp_pix[i,j]
            else:
                std_pix[i,j] = raw_disp_pix[i,j]

    handler_name = "ELAS-CNN-fusion"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    os.makedirs(sgbm_results_fname, exist_ok=True)
    
    img = Image.fromarray(std_pix.astype('uint8'), mode = 'L')
    img.save(sgbm_results_fname + "calc_disp_norm.png", "PNG")
    
    img = Image.fromarray(((std_pix*max_disp)/255).astype('uint8'), mode = 'L')
    img.save(sgbm_results_fname + "calc_disp.png", "PNG")


def elasCNNstd_compute(results_fname, sample_name, max_disp, cosine_weight, window_size, std_th):
    
    image_left_filename  = results_fname + sample_name + "/im0"
    image_right_filename = results_fname + sample_name + "/im1"

    im = Image.open(image_left_filename + ".png")
    im = im.convert("L")
    im.save(image_left_filename + ".pgm")

    im = Image.open(image_right_filename + ".png")
    im = im.convert("L")
    im.save(image_right_filename + ".pgm")

    conv_left = numpy.load("np_data/fst/" + sample_name + "_left_conv.npy")
    conv_right = numpy.load("np_data/fst/" + sample_name + "_right_conv.npy")
    
    raw_disp = Image.open(results_fname + sample_name + "/ConvFastNN/" + sample_name + "_raw_disp.png").convert("L")
    width, height = raw_disp.size
    raw_disp_pix = numpy.atleast_1d(raw_disp).astype('int')
    std_pix = numpy.zeros((height, width))
    for i in range(int(window_size/2), height - int(window_size/2)):
        for j in range(int(window_size/2), width - int(window_size/2)):
            area = int((window_size-1)/2)
            std = raw_disp_pix[i-area:i+area+1,j-area:j+area+1].std()
            if std > std_th:
                std_pix[i,j] = -1
            else:
                std_pix[i,j] = raw_disp_pix[i,j]
    numpy.save("np_data/fst/" + sample_name + "_mask", std_pix)
    mask = numpy.load("np_data/fst/" + sample_name + "_mask.npy")

    conv_left = conv_left.astype(dtype=numpy.float32)
    conv_right = conv_right.astype(dtype=numpy.float32)
    mask = mask.astype(dtype=numpy.float32)

    elasCNNstd.compute(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm",
                       max_disp, 0, cosine_weight, mask, conv_left, conv_right)

    handler_name = "ELAS-CNN-std"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    outDispFileName      = sgbm_results_fname + "calc_disp.png"

    os.makedirs(sgbm_results_fname, exist_ok=True) 

    copyfile("disp.pfm", sgbm_results_fname + "calc_disp.pfm")
    os.remove("disp.pfm")

    disp_pfm = pfm.readPfm(sgbm_results_fname + "calc_disp.pfm")
    cv2.imwrite(outDispFileName, disp_pfm)

    cv2.imwrite(sgbm_results_fname + "calc_disp_norm.png", 255*disp_pfm/max_disp)

