from keras.models import Model
from keras.layers import Conv2D, Dot, Flatten, Input
from PIL import Image
import numpy
import os
import  cv2

patch_size = 9
num_conv = int((patch_size-1)/2)
conv_feature_maps = 112

img = cv2.imread("../../../samples/Middlebury_scenes_2014/trainingQ/Motorcycle/im0.png", 0)

data = numpy.zeros((img.shape[0], img.shape[1], 1))
print(data.shape)

input_layer = Input(shape=data.shape)
conv_layer = input_layer
for i in range(1, num_conv+1):
    conv_layer = Conv2D(conv_feature_maps, kernel_size=3, activation="relu", name="conv"+str(i)) (conv_layer)
    
#flatten_layer = Flatten(name = "flat")(conv_layer)
flatten_layer = conv_layer

model = Model(inputs=input_layer, outputs=flatten_layer)
model.save_weights("w.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# parameter ==========================
wkdir = './'
pb_filename = 'model.pb'

# save model to pb ====================
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

# save keras model as tf pb files ===============
import tensorflow as tf
from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)

# # load & inference the model ==================

from tensorflow.python.platform import gfile
with tf.Session() as sess:
    # load model from pb file
    with gfile.FastGFile(wkdir+'/'+pb_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)
    # write to tensorboard (check tensorboard for each op names)
    writer = tf.summary.FileWriter(wkdir+'/log/')
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()
    # print all operation names 
    print('\n===== ouptut operation names =====\n')
    # for op in sess.graph.get_operations():
    #   print(op)
    # # inference by the model (op name must comes with :0 to specify the index of its output)
    # tensor_output = sess.graph.get_tensor_by_name('import/dense_3/Sigmoid:0')
    # tensor_input = sess.graph.get_tensor_by_name('import/dense_1_input:0')
    # predictions = sess.run(tensor_output, {tensor_input: x})
    # print('\n===== output predicted results =====\n')



