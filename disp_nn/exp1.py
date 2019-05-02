import os
import data
import numpy
from shutil import copyfile
import colorama
from termcolor import colored
from exp.handlers import *

# set log level ('0' - all messages, '1' - no info messages, '2' - no warning messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
colorama.init()

samples_list   = ["Adirondack", "ArtL", "Motorcycle", "Piano", "Recycle", "Shelves", "Teddy"]
samples_fname  = "../samples/Middlebury_scenes_2014/trainingQ/"
#samples_list   = ["Australia","AustraliaP","Bicycle2","Classroom2","Classroom2E","Computer","Crusade","CrusadeP","Djembe","DjembeL","Hoops","Livingroom","Newkuba","Plants","Staircase"]
#samples_fname  = "../samples/Middlebury_scenes_2014/testQ/"

samples_list    = ["Teddy"]
# samples_fname   = "../samples/"
w_full_filename  = "weights/acc1_weights6.h5"
w_fast_filename = "weights/fst1_weights1.h5"
cosine_weight   = 50
support_cosine_weight = 1
support_threshold = 0.87
window_size = 3 #3-7
std_th = 7 #5-20
std_filter = 0
grid_size = 5
results_fname   = "../results/"

for sample_name in samples_list:

    print(colored("-> " + sample_name, 'green'))
    print("-"*50)
    sample_fname   = samples_fname + sample_name
    
    # Create results folder
    os.makedirs(results_fname + sample_name, exist_ok=True) 
    copyfile(sample_fname + "/im0.png", results_fname + sample_name + "/im0.png")
    copyfile(sample_fname + "/im1.png", results_fname + sample_name + "/im1.png")
    copyfile(sample_fname + "/calib.txt", results_fname + sample_name + "/calib.txt")
    if os.path.isfile(sample_fname + "/mask0nocc.png"):
        copyfile(sample_fname + "/mask0nocc.png", results_fname + sample_name + "/mask0nocc.png")
    if os.path.isfile(sample_fname + "/disp0GT.pfm"):
        copyfile(sample_fname + "/disp0GT.pfm", results_fname + sample_name + "/disp0GT.pfm")
    if os.path.isfile(sample_fname + "/disp0.png"):
        copyfile(sample_fname + "/disp0.png", results_fname + sample_name + "/disp0.png")
    #----
    calib_file = open(sample_fname + "/calib.txt")
    max_disp = data.find_max_disp(calib_file)

    #print("handler_name: "+ colored("convFullNN", 'yellow'))
    #convFulNN_compute(results_fname, sample_name, w_full_filename, max_disp)

    #print("handler_name: "+ colored("convFastNN", 'yellow'))
    #convFastNN_compute(results_fname, sample_name, w_fast_filename, max_disp)

    print("handler_name: "+ colored("opencvSGBM", 'yellow'))
    opencvSGBM_compute(results_fname, sample_name, max_disp)

    #print("handler_name: "+ colored("ELAS", 'yellow'))
    #pyelas_compute(results_fname, sample_name, max_disp)

    #print("handler_name: "+ colored("ELAS-CNN", 'yellow'))
    #elasCNN_compute(results_fname, sample_name, w_fast_filename, max_disp, cosine_weight)

    print("handler_name: "+ colored("ELAS-CNN-sup", 'yellow'))
    elasCNNsup_compute(results_fname, sample_name, max_disp, w_fast_filename, cosine_weight,
                       support_cosine_weight, support_threshold, std_filter, window_size, std_th)

    

    calib_file.close()
    
#-----------------------------------------------------------------------------------------------------------------------------------

    #print("handler_name: "+ colored("ELAS-CNN-grid", 'yellow'))
    #elasCNNgrid_compute(results_fname, sample_name, max_disp, w_fast_filename, cosine_weight, grid_size)
    
    #print("handler_name: "+ colored("ELAS-CNN-fusion", 'yellow'))
    #elasCNNfusion_compute(results_fname, sample_name, max_disp, cosine_weight, window_size, std_th)

    #print("handler_name: "+ colored("ELAS-CNN-std", 'yellow'))
    #elasCNNstd_compute(results_fname, sample_name, max_disp, cosine_weight, window_size, std_th)

