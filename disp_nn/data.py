import numpy
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
from PIL import Image, ImageDraw
import time
import matplotlib.pyplot as plt
import cv2
from capi import sgbm as vSGBM


'''
This function gets folder name in /samples/, size of the patch,
high an low thresholds for a negative training sample bias
and disparity scale (e.g., if the disparity image is normalized to 0..255
and maximum disparity is 64 then scale will be 256 / 64 = 4).
The output of this function is three numpy arrays (left input patches,
right input patches and outputs) used for training.
'''
def get_batch(folder_name, patch_size, neg_low, neg_high, scale):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    disp0_pic = Image.open(folder_name + "disp0.png")
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    disp0_pix = numpy.array(disp0_pic)
    width, height = left_pic.size
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)

    left_patches = []
    right_patches = []
    outputs = []
    dataset_size = 0

    for i in range(0, height):
        for j in range(0, width):
            if disp0_pix[i, j] > 0 and j >= (int(disp0_pix[i, j]/scale) + int(patch_size/2)) and j < (width - int(patch_size/2)):
                if i >= int(patch_size/2) and i < (height - int(patch_size/2)):

                    # positive sample
                    left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                           (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                    disp = int(disp0_pix[i, j]/scale)
                    right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                             (j - int(patch_size/2) - disp) : (j + int(patch_size/2) - disp + 1)]
                    left_patches.append(left_patch)
                    right_patches.append(right_patch)
                    outputs.append(1)

                    # negative sample
                    offset = random.randint(neg_low, neg_high)
                    left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                           (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                    right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                             (j - int(patch_size/2) - disp + offset) : (j + int(patch_size/2) - disp + offset + 1)]
                    left_patches.append(left_patch)
                    right_patches.append(right_patch)
                    outputs.append(0)
                    
                    dataset_size += 2
        
    return numpy.array(left_patches), numpy.array(right_patches), numpy.array(outputs)

'''
This function is equal to the upper one but also takes an array of x coordinates which
limit some areas that contain points inappropriate for learning (e.g., occluded).
Such array may look like [10,15,55,75,95,97], which means, that no points with x coordinate
inside [10,15], [55,75] and [95,97] will be in the teaching batch.
'''
def get_batch_areas(folder_name, patch_size, neg_low, neg_high, scale, areas):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    disp0_pic = Image.open(folder_name + "disp0.png")
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    disp0_pix = numpy.atleast_3d(disp0_pic)
    width, height = left_pic.size
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)

    left_patches = []
    right_patches = []
    outputs = []
    dataset_size = 0

    for i in range(0, height):
        for j in range(0, width):
            if disp0_pix[i, j] > 0 and j >= (int(disp0_pix[i, j]/scale) + int(patch_size/2)) and j < (width - int(patch_size/2)):
                if i >= int(patch_size/2) and i < (height - int(patch_size/2)):
                    if numpy.searchsorted(areas, j)%2 == 0: # check if the point is not inside restricted area

                        # positive sample
                        left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                               (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                        disp = int(disp0_pix[i, j]/scale)
                        right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                                 (j - int(patch_size/2) - disp) : (j + int(patch_size/2) - disp + 1)]
                        left_patches.append(left_patch)
                        right_patches.append(right_patch)
                        outputs.append(1)

                        # negative sample
                        offset = random.randint(neg_low, neg_high)
                        left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                               (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
                        right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                                 (j - int(patch_size/2) - disp + offset) : (j + int(patch_size/2) - disp + offset + 1)]
                        left_patches.append(left_patch)
                        right_patches.append(right_patch)
                        outputs.append(0)
                        
                        dataset_size += 2
        
    return numpy.array(left_patches), numpy.array(right_patches), numpy.array(outputs)

'''
This function convolves images and saves them as .npy files.
The function uses full images instead of set of patches. It needs left and right convolutional
subnets as an input.
'''
def convolve_images_ctc(folder_name, patch_size, conv_feature_maps, lctc_model, rctc_model):
    print("begin convolution")
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    border = int(patch_size/2)
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    width, height = left_pic.size
    conv_left_patches = numpy.zeros((height,width,conv_feature_maps))
    conv_right_patches = numpy.zeros((height,width,conv_feature_maps))
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)
    
    timestamp = time.time()
    l_prediction = lctc_model.predict([[norm_left]])
    r_prediction = rctc_model.predict([[norm_right]])
    conv_left_patches[border:-border,border:-border,::] = l_prediction.reshape((height - 2*border, width - 2*border, conv_feature_maps))
    conv_right_patches[border:-border,border:-border,::] = r_prediction.reshape((height - 2*border, width - 2*border, conv_feature_maps))
    print("total time ", round(time.time()-timestamp, 3))
    return conv_left_patches, conv_right_patches

'''
This function takes the result of function "convolve_images_ctc" (convolved full images) and works much faster.
It creates a disparity image and also can save the disparity map before argmax as a .npy.
'''
def disp_map_from_conv_dtc(left_conv, right_conv, patch_size, max_disp, match_th, conv_feature_maps, dtc_model, image_name):
    print("begin disparity computation")
    height = left_conv.shape[0]
    width = right_conv.shape[1]
    disp_pix = numpy.zeros((height,width))
    timestamp = time.time()
    dtc_predictions = dtc_model.predict([numpy.expand_dims(numpy.concatenate((left_conv[::, max_disp:width,::],
                                                                                    right_conv[::,0:width-max_disp,::]), axis = 2), axis = 0)])
    dtc_predictions = numpy.squeeze(dtc_predictions, axis=0)
    for i in range(1, max_disp):
        prediction = dtc_model.predict([numpy.expand_dims(numpy.concatenate((left_conv[::, max_disp:width,::],
                                                                                    right_conv[::,i:width-max_disp+i,::]), axis = 2), axis = 0)])
        dtc_predictions = numpy.concatenate((dtc_predictions, numpy.squeeze(prediction, axis=0)), axis=2)
        print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % (100*(i+1)/max_disp), "%", end = "\r")
    #dtc_predictions[dtc_predictions<=match_th] = 0
    print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % 100, "%", end = "\r")
    numpy.save('np_data/' + image_name + '_predictions', dtc_predictions)
    disp_pix[::, max_disp:width] = (255*(max_disp - numpy.argmax(dtc_predictions, axis = 2)))/max_disp
    print("\ntotal time ", "%.2f" % (time.time()-timestamp))
    img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
    img.save(image_name + ".png", "PNG")

'''
This function takes disparity without argmax (raw prediction of a dense net) and can be used
to implement filters to the disparity ranges in order to create better disparit map or to identify
bad disparity ranges.
'''
def disp_map_from_predict(predictions, left_conv, patch_size, max_disp, match_th, conv_feature_maps, image_name):
    print("begin disparity computation")
    height = left_conv.shape[0]
    width = left_conv.shape[1]
    disp_pix = numpy.zeros((height,width))
    timestamp = time.time()
    sorted_predictions = numpy.sort(predictions)
    for i in range(height):
        for j in range(width-max_disp):
            if sorted_predictions[i,j,-1] - sorted_predictions[i,j,-10:-1].mean() < match_th:
                disp_pix[i,j+max_disp] = max_disp
            else:
                disp_pix[i,j+max_disp] = numpy.argmax(predictions[i,j])
    #dtc_predictions[dtc_predictions<=match_th] = 0
    print("\rtime ", "%.2f" % (time.time()-timestamp), " progress ", "%.2f" % 100, "%", end = "\r")
    #numpy.save('np_data/' + image_name + '_predictions', dtc_predictions)
    disp_pix[::,max_disp:width] = (255*(max_disp - disp_pix[::,max_disp:width]))/max_disp
    print("\ntotal time ", "%.2f" % (time.time()-timestamp))
    img = Image.fromarray(disp_pix.astype('uint8'), mode = 'L')
    img.save(image_name + ".png", "PNG")

'''
!!! All the below functions save their results in /work directory. !!!
'''

def sgbm(predictions, max_disp, cross_size):
    """Perform SGBM"""
    
    print("SGBM computation")

    predictions = 1 - predictions # convert predictions to matching cost

    timestamp = time.time()
    disp_pix = vSGBM.compute(predictions, cross_size, 0)

    
    disp_pix[:, max_disp:] = max_disp - disp_pix[:, max_disp:]
    #disp_pix[:, max_disp:] = (255*disp_pix[:, max_disp:])/max_disp
    print("\ntotal time ", "%.2f" % (time.time()-timestamp))

    img = disp_pix.astype('uint8')
    disp_pix = cv2.bilateralFilter(img, 5, 5, 5)

    return disp_pix
    
    #return Image.fromarray(disp_pix.astype('uint8'), mode = 'L')

'''
This function takes a ground truth disparity image and a real disparity image and can be used
to create graphs and histograms to learn different metrics that can help to identify bad pixels (e.g., noisy).
The implementation shown creates a metric of standart deviation of set of absolute differences between
the pixel value and its neighbours inside a window.
'''
def comp_metric_corr(name1, name2, predictions, patch_size, max_disp, error_threshold):
    disp_ref = Image.open(name1 + ".png")
    disp = Image.open(name2 + ".png")
    width, height = disp.size
    disp_ref_pix = numpy.atleast_1d(disp_ref)
    disp_pix = numpy.atleast_1d(disp).astype('int')
    err_arr = []
    std_arrG = []
    std_arrB = []
    
    pix_num = (height - patch_size + 1) * (width - patch_size - max_disp + 1)
    error_num = 0
    not_recognized = 0
    n_bins = 20
    sorted_predictions = numpy.sort(predictions)

    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            #sad = 0
            sad = []
            area = 1 # changes window size, area=1 -> window=3, area=2 -> window=5 etc.
            #for pix in numpy.nditer(disp_pix[i-area:i+area+1,j-area:j+area+1]):
                #sad += abs(disp_pix[i,j]-pix)
            sad = (abs(disp_pix[i,j] - disp_pix[i-area:i+area+1,j-area:j+area+1])).std()
                
            if int(disp_ref_pix[i,j]) == 0:
                not_recognized += 1
                
            elif abs(int(disp_ref_pix[i,j]) - int(disp_pix[i,j])) > error_threshold:
                std_arrB.append(sad)
                #std_arrB.append(sorted_predictions[i,j-max_disp,-1] - sorted_predictions[i,j-max_disp,-2])
                #std_arrB.append(sorted_predictions[i,j-max_disp,-2:-1].std())
                #std_arrB.append(sorted_predictions[i,j-max_disp,-1] - sorted_predictions[i,j-max_disp,-10:-2].mean())
            else:
                std_arrG.append(sad)
                #std_arrG.append(sorted_predictions[i,j-max_disp,-1] - sorted_predictions[i,j-max_disp,-2])
                #std_arrG.append(sorted_predictions[i,j-max_disp,-2:-1].std())
                #std_arrG.append(sorted_predictions[i,j-max_disp,-1] - sorted_predictions[i,j-max_disp,-10:-2].mean())
    print(sad)

    plt.hist(std_arrG, n_bins, histtype='step', label='good',fill=False)
    plt.hist(std_arrB, n_bins, histtype='step', label='bad',fill=False)
    plt.legend()
    plt.savefig("work/hist_diff_std_3_snowflake.png")

'''
Takes a disparity image and substitutes with zeros all the pixels for which the SAD metric inside of the window
exceeds the threshold value.
'''
def sad_filter(name2, patch_size, max_disp, error_threshold, window_size):
    disp = Image.open(name2 + ".png")
    width, height = disp.size
    disp_pix = numpy.atleast_1d(disp).astype('int')
    sad_pix = numpy.zeros((height, width))

    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            sad=0
            area = int((window_size-1)/2)
            for pix in numpy.nditer(disp_pix[i-area:i+area+1,j-area:j+area+1]):
                sad += abs(disp_pix[i,j]-pix)
            if sad < error_threshold:
                sad_pix[i,j]=disp_pix[i,j]
                
    img = Image.fromarray(sad_pix.astype('uint8'), mode = 'L')
    img.save("work/sad_disp" + str(window_size) + '_' + str(error_threshold) + ".png", "PNG")

'''
Takes a disparity image and substitutes with zeros all the pixels for which the STD (standard deviation)
metric inside of the window exceeds the threshold value.
'''
def std_filter(name2, patch_size, max_disp, error_threshold, window_size):
    disp = Image.open(name2 + ".png")
    width, height = disp.size
    disp_pix = numpy.atleast_1d(disp).astype('int')
    std_pix = numpy.zeros((height, width))

    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            area = int((window_size-1)/2)
            std = disp_pix[i-area:i+area+1,j-area:j+area+1].std()
            if std < error_threshold:
                std_pix[i,j]=disp_pix[i,j]
                
    img = Image.fromarray(std_pix.astype('uint8'), mode = 'L')
    img.save("work/std_disp" + str(window_size) + '_' + str(error_threshold) + ".png", "PNG")

'''
Takes two disparity images, ground truth and real, and compares them to calculate the error rate
using the error threshold. Also saves a "filtered_disparity" - real disparity with only good pixels kept.
'''
def comp_error_in_area(name1, name2, patch_size, max_disp, error_threshold):
    disp_ref = Image.open(name1 + ".png")
    disp = Image.open(name2 + ".png")
    width, height = disp.size
    disp_ref_pix = numpy.atleast_1d(disp_ref)
    disp_pix = numpy.atleast_1d(disp)
    filtered_pix = numpy.zeros((height,width))
    
    pix_num = (height - patch_size + 1) * (width - patch_size - max_disp + 1)
    error_num = 0
    not_recognized = 0

    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            if int(disp_ref_pix[i,j]) == 0:
                not_recognized += 1
            elif abs(int(disp_ref_pix[i,j]) - int(disp_pix[i,j])) > error_threshold:
                error_num += 1
            else:
                filtered_pix[i, j] = disp_pix[i, j]
    print("error rate ", round(error_num*100/(pix_num - not_recognized), 2))
    print("not recognized", not_recognized)
    print("num of pixels", pix_num)
    print("mum of errors", error_num)
    img = Image.fromarray(filtered_pix.astype('uint8'), mode = 'L')
    img.save("work/no_error_disp_" + str(error_threshold) + ".png", "PNG")

'''
Restores a disparity image drom a filtered disparity image obtained with sad_filter or std_filter.
Finds left and right pixels of a black (bad) pixel and fulls the gap with a linear distribution.
'''
def approx_disp(image, patch_size, max_disp):
    disp_pic = Image.open(image + ".png")
    disp_pix = numpy.atleast_1d(disp_pic)
    width, height = disp_pic.size
    filtered_pix = disp_pix.copy()
    border = 10
    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(max_disp + int(patch_size/2), width - int(patch_size/2)):
            if filtered_pix[i,j] == 0:
                left = filtered_pix[i,j-1]
                right = 0
                pos = 1
                while right == 0:
                    right = filtered_pix[i,j+pos]
                    pos+=1
                    if pos > width - int(patch_size/2) - j:
                        break
                if left > 0 and right > 0:
                    if pos < border:
                        if left == right:
                            filtered_pix[i, j:j+pos] = numpy.full((pos),right)
                        else:
                            filtered_pix[i, j:j+pos] = numpy.linspace(left, right, pos+1, endpoint=False)[1:]
                    else:
                        filtered_pix[i, j:j+border] = numpy.full((border),left)
                elif left == 0:
                    filtered_pix[i, j:j+pos] = numpy.full((pos),right)
                else:
                    filtered_pix[i, j:j+pos] = numpy.full((pos),left)

    img = Image.fromarray(filtered_pix.astype('uint8'), mode = 'L')
    img.save("work/approx_disp" + ".png", "PNG")

'''
Restores a disparity image drom a filtered disparity image obtained with sad_filter or std_filter.
Finds left, right, top and bottom pixels of a black (bad) pixel and restores its value according to
the values of the nearest meaning neighbours.
'''
def approx_disp_full(image, patch_size, max_disp):
    disp_pic = Image.open(image + ".png")
    disp_pix = numpy.atleast_1d(disp_pic).astype('int')
    width, height = disp_pic.size
    filtered_pix = disp_pix.copy()
    h_border = 10
    v_border = 10
    for i in range(int(patch_size/2) + 1, height - int(patch_size/2) - 1):
        for j in range(max_disp + int(patch_size/2) + 1, width - int(patch_size/2) - 1):
            if filtered_pix[i,j] == 0:
                left = filtered_pix[i,j-1]
                top = filtered_pix[i-1,j]
                right = 0
                bottom = 0
                h_pos = 1
                v_pos = 1
                h_pix = 0
                v_pix = 0
                while right == 0:
                    right = filtered_pix[i,j+h_pos]
                    h_pos+=1
                    if h_pos > width - int(patch_size/2) - j:
                        break
                while bottom == 0:
                    right = filtered_pix[i+v_pos,j]
                    v_pos+=1
                    if v_pos > height - int(patch_size/2) - i:
                        break
                if left > 0 and right > 0:
                    if h_pos < h_border:
                        if left == right:
                            h_pix = right
                        else:
                            h_pix = left + (right-left)/h_pos
                    else:
                        h_pix = left
                elif left == 0 and right > 0:
                    h_pix = right
                elif right == 0 and left > 0:
                    h_pix = left
                    
                if top > 0 and bottom > 0:
                    if v_pos < v_border:
                        if top == bottom:
                            v_pix = bottom
                        else:
                            v_pix = top + (bottom-top)/v_pos
                    else:
                        v_pix = top
                elif top == 0 and bottom > 0:
                    v_pix = bottom
                elif bottom == 0 and top > 0:
                    v_pix = top

                #if h_pix > 0 and v_pix > 0:
                    #filtered_pix[i,j] = (h_pix + v_pix)/2
                if h_pix > 0:
                    filtered_pix[i,j] = h_pix
                elif h_pix == 0 and v_pix > 0:
                    filtered_pix[i,j] = v_pix
                #elif v_pix == 0 and h_pix > 0:
                    #filtered_pix[i,j] = h_pix

    img = Image.fromarray(filtered_pix.astype('uint8'), mode = 'L')
    img.save("work/approx_disp_full" + ".png", "PNG")
