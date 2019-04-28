from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
import numpy
import data
import matplotlib.pyplot as plt

# constants
training_size = 576000
conv_feature_maps = 112
dense_size = 384
patch_size = 11
max_disp = 64
image_name = "cones"
scale = 4
neg_high = 6
neg_low = 3
areas = [0,88,234,291,410,449]
num_of_batches = 128
num_of_epochs = 10

# fix random seed for reproducibility
numpy.random.seed(7)

# load dataset
left1, right1, outputs1 = data.get_batch("../samples/" + image_name + "/", patch_size, neg_low, neg_high, scale)
left2, right2, outputs2 = data.get_batch("../samples/" + image_name + "/", patch_size, neg_low, neg_high, scale)

left = numpy.concatenate((left1,left2))
right = numpy.concatenate((right1,right2))
outputs = numpy.concatenate((outputs1,outputs2))

print(outputs.shape)

left = left[0:training_size]
right = right[0:training_size]
outputs = outputs[0:training_size]

# create model
left_input = Input(shape=(patch_size, patch_size, 1, ))
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc1") (left_input)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc2") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc3") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc4") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="lc5") (left_conv)
left_flatten = Flatten(name = "left_flatten_layer")(left_conv)

right_input = Input(shape=(patch_size, patch_size, 1, ))
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc1") (right_input)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc2") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc3") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc4") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="rc5") (right_conv)
right_flatten = Flatten(name = "right_flatten_layer")(right_conv)

conc_layer = Concatenate(name = "d1")([left_flatten, right_flatten])
dense_layer = Dense(dense_size, activation="relu", name = "d2")(conc_layer)
dense_layer = Dense(dense_size, activation="relu", name = "d3")(dense_layer)
output_layer = Dense(1, activation="sigmoid", name = "d4")(dense_layer)
#output_layer = Dot(axes=-1, normalize = True)([left_flatten, right_flatten])

model = Model(inputs=[left_input, right_input], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit([left,right], outputs, epochs=num_of_epochs, batch_size=num_of_batches)
predictions = model.predict([left[0:10], right[0:10]])
print(predictions)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#plot_model(model, show_shapes=True, to_file='model.png')
model.save_weights("weights/acc1_weights6.h5")
########################################################################################

