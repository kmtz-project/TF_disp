from nn.convFullNN import ConvFullNN
from nn.convFastNN import ConvFastNN
import cv2
import os
import numpy
from PIL import Image
from shutil import copyfile
from utils import pfm
import data

from capi import pyelas, elasCNN, elasCNNsup, elasCNNgrid

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
        if max_disp>8:
            sgbm_max_disp = 16*(int(max_disp/16)+1)
        else:
            sgbm_max_disp = 16*(int(max_disp/16))
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

def elasCNN_compute(results_fname, sample_name, w_filename, max_disp, cosine_weight):
    
    image_left_filename  = results_fname + sample_name + "/im0"
    image_right_filename = results_fname + sample_name + "/im1"

    im = Image.open(image_left_filename + ".png")
    im = im.convert("L")
    im.save(image_left_filename + ".pgm")

    im = Image.open(image_right_filename + ".png")
    im = im.convert("L")
    im.save(image_right_filename + ".pgm")

    # Get convolution
    handler_name = "ELAS-CNN"
    exists = os.path.isfile("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv.npy")
    ctc_width, ctc_height = im.size
    if not exists:
        dnet = ConvFastNN("ELAS-CNN", results_fname)
        dnet.max_disp = max_disp
        dnet.createResultDir(sample_name)
        dnet.createCTCModels(ctc_height, ctc_width)
        dnet.loadCTCWeights(w_filename)
        dnet.convolve_images_ctc(results_fname + sample_name)
        numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv",dnet.conv_left_patches)
        numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv",dnet.conv_right_patches)
        del dnet
    # ------------------------

    conv_left = numpy.load("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv.npy")
    conv_right = numpy.load("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv.npy")

    conv_left = conv_left.astype(dtype=numpy.float32)
    conv_right = conv_right.astype(dtype=numpy.float32)

    grid = elasCNN.compute(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm", max_disp, 0,
                           cosine_weight, conv_left, conv_right)
    
    handler_name = "ELAS-CNN"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    outDispFileName      = sgbm_results_fname + "calc_disp.png"

    os.makedirs(sgbm_results_fname, exist_ok=True) 

    copyfile("disp.pfm", sgbm_results_fname + "calc_disp.pfm")
    os.remove("disp.pfm")

    disp_pfm = pfm.readPfm(sgbm_results_fname + "calc_disp.pfm")
    cv2.imwrite(outDispFileName, disp_pfm)

    cv2.imwrite(sgbm_results_fname + "calc_disp_norm.png", 255*disp_pfm/max_disp)

    grid_step = 5
    error_th = 2
    im = Image.open(image_left_filename + ".png")
    grid_pix = numpy.atleast_3d(im.convert("L").convert("RGB")).copy()
    exists = os.path.isfile("../results/" + sample_name + "/disp0GT.pfm")
    if exists:
        disp_ref_pix = pfm.readPfm("../results/" + sample_name + "/disp0GT.pfm")
    else:
        im = Image.open("../results/" + sample_name + "/disp0.png")
        disp_ref_pix = numpy.atleast_1d(im.convert("L")).copy()
    grid_name = "../results/" + sample_name + "/" + handler_name + "/grid.png"
    data.comp_grid_error(grid,grid_pix,disp_ref_pix,error_th,grid_step,grid_name)

def elasCNNsup_compute(results_fname, sample_name, max_disp, w_filename, cosine_weight,
                       support_cosine_weight, support_threshold, std_filter, window_size, std_th):
    
    image_left_filename  = results_fname + sample_name + "/im0"
    image_right_filename = results_fname + sample_name + "/im1"

    im = Image.open(image_left_filename + ".png")
    im = im.convert("L")
    im.save(image_left_filename + ".pgm")

    im = Image.open(image_right_filename + ".png")
    im = im.convert("L")
    im.save(image_right_filename + ".pgm")

    # Get convolution
    handler_name = "ELAS-CNN-sup"

    exists = os.path.isfile("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv.npy")
    ctc_width, ctc_height = im.size
    if not exists:
        dnet = ConvFastNN("ELAS-CNN-sup", results_fname)
        dnet.max_disp = max_disp
        dnet.createResultDir(sample_name)
        dnet.createCTCModels(ctc_height, ctc_width)
        dnet.loadCTCWeights(w_filename)
        dnet.convolve_images_ctc(results_fname + sample_name)
        numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv",dnet.conv_left_patches)
        numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv",dnet.conv_right_patches)
        del dnet
    # ------------------------

    conv_left = numpy.load("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv.npy")
    conv_right = numpy.load("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv.npy")

    conv_left = conv_left.astype(dtype=numpy.float32)
    conv_right = conv_right.astype(dtype=numpy.float32)

    if std_filter == 1:
        mask_name = "../results/" + sample_name + "/w" + str(window_size) + "_th" + str(std_th) + "_mask"
        exists = os.path.isfile(mask_name + ".npy")

        if not exists:
            if os.path.isfile(results_fname + sample_name + "/ConvFastNN/" + sample_name + "_raw_disp.png"):
                raw_disp = Image.open(results_fname + sample_name + "/ConvFastNN/" + sample_name + "_raw_disp.png").convert("L")
            else:
                print("Please run ConvFastNN first")
                exit()
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
            numpy.save(mask_name, std_pix)
        mask = numpy.load(mask_name + ".npy")
    else:
        mask = numpy.zeros((ctc_height,ctc_width))

    grid = elasCNNsup.compute(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm",
                       max_disp, 0, support_threshold, cosine_weight, std_filter, support_cosine_weight, mask, conv_left, conv_right)

    handler_name = "ELAS-CNN-sup"
    sgbm_results_fname   = results_fname + sample_name + "/" + handler_name + "/"
    outDispFileName      = sgbm_results_fname + "calc_disp.png"

    os.makedirs(sgbm_results_fname, exist_ok=True) 

    copyfile("disp.pfm", sgbm_results_fname + "calc_disp.pfm")
    os.remove("disp.pfm")

    disp_pfm = pfm.readPfm(sgbm_results_fname + "calc_disp.pfm")
    cv2.imwrite(outDispFileName, disp_pfm)

    cv2.imwrite(sgbm_results_fname + "calc_disp_norm.png", 255*disp_pfm/max_disp)

    grid_step = 5
    error_th = 2
    im = Image.open(image_left_filename + ".png")
    grid_pix = numpy.atleast_3d(im.convert("L").convert("RGB")).copy()
    exists = os.path.isfile("../results/" + sample_name + "/disp0GT.pfm")
    if exists:
        disp_ref_pix = pfm.readPfm("../results/" + sample_name + "/disp0GT.pfm")
    else:
        im = Image.open("../results/" + sample_name + "/disp0.png")
        disp_ref_pix = numpy.atleast_1d(im.convert("L")).copy()
    grid_name = "../results/" + sample_name + "/" + handler_name + "/grid.png"
    data.comp_grid_error(grid,grid_pix,disp_ref_pix,error_th,grid_step,grid_name)

    
# different evaluation methods below
# ----------------------------------------------------------------------------------------------------------------------------------
def elasCNNgrid_compute(results_fname, sample_name, max_disp, w_filename, cosine_weight, grid_size):
    
    image_left_filename  = results_fname + sample_name + "/im0"
    image_right_filename = results_fname + sample_name + "/im1"

    im = Image.open(image_left_filename + ".png")
    im = im.convert("L")
    im.save(image_left_filename + ".pgm")

    im = Image.open(image_right_filename + ".png")
    im = im.convert("L")
    im.save(image_right_filename + ".pgm")

    handler_name = "ELAS-CNN-grid"
    ctc_width, ctc_height = im.size
    exists = os.path.isfile("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv.npy")
    if not exists:
    # Store configuration file values

        dnet = ConvFastNN("ELAS-CNN-grid", results_fname)
        dnet.max_disp = max_disp

        dnet.createResultDir(sample_name)

        dnet.createCTCModels(ctc_height, ctc_width)
        dnet.loadCTCWeights(w_filename)
        dnet.convolve_images_ctc(results_fname + sample_name)
        numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv",dnet.conv_left_patches)
        numpy.save("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv",dnet.conv_right_patches)
        del dnet
    # ------------------------

    conv_left = numpy.load("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_left_conv.npy")
    conv_right = numpy.load("../results/" + sample_name + "/" + handler_name + "/" + sample_name + "_right_conv.npy")

    # Create GT array
    folder_name = results_fname + sample_name
    occ_pic = Image.open(folder_name + "/mask0nocc.png").convert("L")
    disp0_pix = pfm.readPfm(folder_name + "/disp0GT.pfm")
    occ_pix = numpy.atleast_3d(occ_pic)
    GT_arr = disp0_pix.copy()
    
    for i in range(ctc_height):
        for j in range(ctc_width):
            if occ_pix[i,j] < 255:
                GT_arr[i,j] = -1

    conv_left = conv_left.astype(dtype=numpy.float32)
    conv_right = conv_right.astype(dtype=numpy.float32)
    GT_arr = GT_arr.astype(dtype=numpy.float32)

    elasCNNgrid.compute(image_left_filename + ".pgm", image_right_filename + ".pgm", "disp.pfm",
                       max_disp, 0, cosine_weight, grid_size, GT_arr, conv_left, conv_right)

    handler_name = "ELAS-CNN-grid"
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

