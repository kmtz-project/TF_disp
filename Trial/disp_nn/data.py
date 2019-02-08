import numpy
import random
from PIL import Image, ImageDraw

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

def get_random_sample(folder_name, patch_size, neg_low, neg_high, scale, pos_or_neg, i, j):
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

    if disp0_pix[i, j] > 0:
        left_patch = []
        right_patch = []
        disp = int(disp0_pix[i, j]/scale)
        if pos_or_neg == 1:
            left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                   (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
            right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                     (j - int(patch_size/2) - disp) : (j + int(patch_size/2) - disp + 1)]
        else:
            offset = random.randint(neg_low, neg_high)
            left_patch = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                   (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
            right_patch = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                     (j - int(patch_size/2) - disp + offset) : (j + int(patch_size/2) - disp + offset + 1)]
        
        return numpy.array(left_patch), numpy.array(right_patch)
    else:
        print("disparity at this point equals 0")

def get_image_in_patches(folder_name, patch_size):
    left_pic = Image.open(folder_name + "im0.png").convert("L")
    right_pic = Image.open(folder_name + "im1.png").convert("L")
    
    left_pix = numpy.atleast_3d(left_pic)
    right_pix = numpy.atleast_3d(right_pic)
    width, height = left_pic.size
    
    left_f = lambda x: (x - left_pix.mean())/left_pix.std()
    norm_left = left_f(left_pix)
    right_f = lambda x: (x - right_pix.mean())/right_pix.std()
    norm_right = right_f(right_pix)

    left_patches = numpy.zeros((height,width, patch_size, patch_size, 1))
    right_patches = numpy.zeros((height,width, patch_size, patch_size, 1))
    dataset_size = 0

    for i in range(int(patch_size/2), height - int(patch_size/2)):
        for j in range(int(patch_size/2), width - int(patch_size/2)):

            left_patches[i, j, ::, ::, ::] = norm_left[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                                       (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
            right_patches[i, j, ::, ::, ::] = norm_right[(i - int(patch_size/2)) : (i + int(patch_size/2) + 1),
                                                         (j - int(patch_size/2)) : (j + int(patch_size/2) + 1)]
        
    return left_patches, right_patches

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
    img.save("filtered_disp_" + str(error_threshold) + ".png", "PNG")

patch_size = 11
max_disp = 63
error_threshold = 12
comp_error_in_area("../samples/cones/disp0", "../samples/cones/disp", patch_size, max_disp, error_threshold)
            

#left, right = get_batch_from_image("../samples/cones/", 11, 63)
#left, right, outputs = get_batch("../samples/cones/", 11, 3, 6, 4)
#print(left.shape)
#print(left_patches[0])
#print(right_patches[0])
#print(left_patches[1])
#print(right_patches[1])
