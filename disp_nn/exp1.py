import os
import data
import numpy
from shutil import copyfile
import colorama
from termcolor import colored
from exp.routine import *

# set log level ('0' - all messages, '1' - no info messages, '2' - no warning messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
colorama.init()

samples_list   = ["Adirondack", "ArtL", "Motorcycle", "Piano", "Recycle", "Shelves", "Teddy"]
samples_fname  = "../samples/Middlebury_scenes_2014/trainingQ/"
# samples_list    = ["pattern1"]
# samples_fname   = "../samples/"
w_ful_filename  = "weights/acc1_weights6.h5"
w_fast_filename = "weights/fst1_weights1.h5"
max_disp        = 64
results_fname   = "../results/"

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
    convFulNN_compute(results_fname, sample_name, w_ful_filename, max_disp)

    print("handler_name: "+ colored("convFastNN", 'yellow'))
    convFastNN_compute(results_fname, sample_name, w_fast_filename, max_disp)

    print("handler_name: "+ colored("opencvSGBM", 'yellow'))
    opencvSGBM_compute(results_fname, sample_name, max_disp)

    print("handler_name: "+ colored("ELAS", 'yellow'))
    elas_compute(results_fname, sample_name, max_disp)

