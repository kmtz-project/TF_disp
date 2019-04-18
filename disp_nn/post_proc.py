import data
import utils
import numpy

# constants
patch_size = 11
image_name = "kron"
error_threshold = 12
max_disp = 60
cpu_only = True
conv_feature_maps = 112
match_th = 0.3
sad_threshold = 800
window_size = 3
std_threshold = 5

P1 = 5
P2 = 200 

# set GPU use
#if cpu_only:
#    utils.set_gpu(False)

#data.std_filter("work" + "/disp", patch_size, max_disp, std_threshold, window_size)    
#data.approx_disp_full("work/" + "/std_disp3_5", patch_size, max_disp)

#data.comp_error_in_area("work" + "/norm_disp", "work" + "/sgbm_disp", patch_size, max_disp, error_threshold)
#left = numpy.load('np_data/' + image_name + '_left_conv.npy')
predictions = numpy.load('np_data/' + image_name + '_disp_dtc_predictions.npy')
#data.comp_metric_corr("work" + "/norm_disp", "work" + "/disp", predictions, patch_size, max_disp, error_threshold)
#data.disp_map_from_predict(predictions, left, patch_size, max_disp, match_th, conv_feature_maps, image_name)
#data.sad_filter("work" + "/cones_disp", patch_size, max_disp, sad_threshold, window_size)

img = data.sgbm(predictions, max_disp, P1, P2)
img.save("work/" + image_name + "_sgbm_c.png", "PNG")


