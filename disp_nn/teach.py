from keras.utils import plot_model
from keras.models import Sequential, Model
from nn.convFullNN import ConvFullNN
from nn.convFastNN import ConvFastNN
import numpy
import data
import matplotlib.pyplot as plt

# constants
samples_list   = ["Adirondack", "ArtL", "Motorcycle", "Piano", "Recycle", "Shelves", "Teddy"]
samples_fname  = "../samples/Middlebury_scenes_2014/trainingQ/"
training_size = 64000 #576000
sample_size_limit = 10000 #100000
conv_feature_maps = 112
dense_size = 384
patch_size = 9
dense_num = 4
scale = 1
neg_high = 6
neg_low = 3
num_of_batches = 128
num_of_epochs = 20
weights_name = "fw-s"
nn_fast = True

def teach_nn(nn_fast,samples_list,samples_fname,training_size,sample_size_limit,conv_feature_maps,dense_size,
             patch_size,dense_num,scale,neg_high,neg_low,num_of_batches,num_of_epochs,weights_name):
    # fix random seed for reproducibility
    numpy.random.seed(7)

    left = numpy.empty((0,patch_size,patch_size,1))
    right = numpy.empty((0,patch_size,patch_size,1))
    outputs = numpy.empty((0,))

    # load dataset
    for sample in samples_list:
            left_arr, right_arr, outputs_arr = data.get_batch_occ(samples_fname + sample + "/", patch_size, neg_low, neg_high, scale)
            left_arr = left_arr[0:sample_size_limit]
            right_arr = right_arr[0:sample_size_limit]
            outputs_arr = outputs_arr[0:sample_size_limit]
            left = numpy.concatenate((left,left_arr),axis=0)
            right = numpy.concatenate((right,right_arr),axis=0)
            outputs = numpy.concatenate((outputs,outputs_arr),axis=0)
            
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(left)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(right)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(outputs)
    print(left.shape)

    left = left[0:training_size]
    right = right[0:training_size]
    outputs = outputs[0:training_size]

    if nn_fast:
        fast_net = ConvFastNN("convFastNN", "")
        fast_net.conv_feature_maps = conv_feature_maps
        fast_net.dense_size = dense_size
        fast_net.patch_size = patch_size
        fast_net.createFastModel()
        model = fast_net.fast_model
    else:
        full_net = ConvFullNN("convFullNN", "")
        full_net.conv_feature_maps = conv_feature_maps
        full_net.dense_size = dense_size
        full_net.dense_num = dense_num
        full_net.patch_size = patch_size
        full_net.createFullModel()
        model = full_net.full_model

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit([left,right], outputs, epochs=num_of_epochs, batch_size=num_of_batches)
    predictions = model.predict([left[0:10], right[0:10]])
    print(predictions, outputs[0:10])

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("img/model/acc_" + weights_name + ".png")
    plt.clf()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("img/model/loss_" + weights_name + ".png")

    #plot_model(model, show_shapes=True, to_file='model.png')
    model.save_weights("weights/" + weights_name + ".h5")
    
#--------------------------------------------------------------------------------------------------------
    
teach_nn(nn_fast,samples_list,samples_fname,training_size,sample_size_limit,conv_feature_maps,dense_size,
             patch_size,dense_num,scale,neg_high,neg_low,num_of_batches,num_of_epochs,weights_name)
