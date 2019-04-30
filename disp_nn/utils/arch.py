import tensorflow as tf
from keras import backend as K

# turn on GPU
def set_gpu(GPU):
    if GPU:
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 1
        num_GPU = 0
    num_cores = 8

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)
	
	
def set_cpu_cores(num_cpu, num_cores):

	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_cpu, 'GPU' : 0})
    
	session = tf.Session(config=config)
	K.set_session(session)

