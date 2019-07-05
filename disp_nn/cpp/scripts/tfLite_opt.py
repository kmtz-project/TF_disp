import tensorflow as tf

saved_model_file = "model_3x3x10_r.h5"

converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_file)
tflite_model = converter.convert()
open("lmodel_3x3x10_r.tflite", "wb").write(tflite_model)

