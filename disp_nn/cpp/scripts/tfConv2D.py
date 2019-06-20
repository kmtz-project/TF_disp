from keras.models import Model
from keras.layers import Conv2D, Dot, Flatten, Input
from PIL import Image
import numpy
import os
import  cv2

num_conv = 1
conv_feature_maps = 10

data = numpy.zeros((3, 3, 1))
print(data.shape)

input_layer = Input(shape=data.shape)
conv_layer = input_layer
for i in range(1, num_conv+1):
    conv_layer = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="conv"+str(i)) (conv_layer)
    
#flatten_layer = Flatten(name = "flat")(conv_layer)
flatten_layer = conv_layer

model = Model(inputs=input_layer, outputs=flatten_layer)
model.save_weights("w3x3x" + str(conv_feature_maps) + ".h5")
model_json = model.to_json()
with open("model_3x3x" + str(conv_feature_maps) + ".json", "w") as json_file:
    json_file.write(model_json)

