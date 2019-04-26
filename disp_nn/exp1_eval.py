import os
#import data
import numpy as np
from shutil import copyfile
from utils import pfm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import colorama
from termcolor import colored

# set log level ('0' - all messages, '1' - no info messages, '2' - no warning messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
colorama.init()

#samples_fname  = "../samples/Middlebury_scenes_2014/trainingQ/"
#samples_list   =  ["Adirondack", "ArtL", "Motorcycle", "Piano", "Recycle", "Shelves", "Teddy",]
samples_fname  = "../samples/Middlebury_scenes_2014/testQ/"
samples_list   = ["Australia","AustraliaP","Bicycle2","Classroom2","Classroom2E","Computer","Crusade","CrusadeP","Djembe","DjembeL","Hoops","Livingroom","Newkuba","Plants","Staircase"]
#samples_list   = ["Teddy"]
#samples_fname  = "../samples/"

handlers_list   = ["convFullNN", "convFastNN", "opencvSGBM", "ELAS", "ELAS-CNN", "ELAS-CNN-sup"]#, "ELAS-CNN-fusion", "ELAS-CNN-std"]
#handlers_list   = ["ELAS-CNN-std"]
file = open("evaluation_test.txt", "w")

for sample_name in samples_list:
    print(colored("-> " + sample_name, 'green'))
    print("-"*50)
    file.write("-> " + sample_name + "\n")
    file.write("-"*50 + "\n")
    sample_fname   = samples_fname + sample_name
    golden_disp_fname = samples_fname + sample_name + "/disp0GT.pfm"

    for handler_name in handlers_list:
        calc_disp_fname   = "../results/" + sample_name + "/" + handler_name + "/calc_disp.png"
        max_disp = 64

        golden_disp = pfm.readPfm(golden_disp_fname)
        calc_disp   = cv2.imread(calc_disp_fname, 0)

        diff = np.absolute(calc_disp - golden_disp)

        cv2.imwrite("../results/" + sample_name + "/" + handler_name + "/golden_disp.png", golden_disp)

        #err_rate = [np.sum(diff[:, max_disp:] > i)/diff.size for i in range(60)]
        error =  np.sum(diff[:, max_disp:] > 2)/diff.size

        print("handler_name: "+ colored(handler_name, 'yellow'))
        print(colored("err_rate = ", "red"), '{:.4f}'.format(error))
        print()

        file.write("handler_name: " + handler_name + "\n")
        file.write("err_rate = " + str(round(error,4)) + "\n")
        file.write("\n")

        calc_disp = calc_disp.astype("int32")
        calc_disp[:, max_disp:] = (255*calc_disp[:, max_disp:])/max_disp
        cv2.imwrite("../results/" + sample_name + "/" + handler_name + "/calc_disp_norm.png", calc_disp)

file.close()


