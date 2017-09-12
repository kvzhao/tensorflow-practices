import numpy as np
import tensorflow as tf

from pprint import pprint
print (tf.__version__)
# my tf shows 1.3.0-dev20170828

import tensorflow.contrib.rnn as rnn

from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput
from tensorflow.python.layers.core import Dense

sequence_length = [3, 4, 3, 1, 0]
batch_size = 5
max_time = 8
input_size = 7
hidden_size = 10
output_size = 3

inputs = np.random.randn(batch_size, max_time, input_size).astype(np.float32)
output_layer = Dense(output_size)

print (' --- output layer (Dense) --- ')
pprint (output_layer.__dict__)
print ('Output layer has no varialbes: _trainable_variables')

decoder_cell = rnn.BasicRNNCell(hidden_size)

# contrib.seq2seq training helper
helper = TrainingHelper(inputs, sequence_length)

# helper serves an input of Basic Decoder
decoder = BasicDecoder(
    cell = decoder_cell,
    helper=helper,
    initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
    output_layer=output_layer
)

print (decoder)
pprint (decoder.__dict__)

attr_names =  [attr for attr in dir(decoder) if '__' not in attr]
pprint (attr_names)

print (decoder.step)
print(decoder.output_size)
print(decoder.output_dtype)
print(decoder.batch_size)

print ('--- initialize decoder ---')

'''
    note: after initialize, the follwing error will disappear
        ERROR:tensorflow:==================================
        Object was never used (type <class 'tensorflow.python.ops.tensor_array_ops.TensorArray'>):
'''
first_finished, first_inputs, first_state = decoder.initialize()
print (first_finished)
print (first_inputs)
print (first_state)

print ('--- Unroll single step ---')

## UNROLL SINGLE STEP
step_outputs, step_state, step_next_inputs, step_finished = decoder.step(
    time=tf.constant(0),
    inputs=first_inputs,
    state=first_state
)

print (step_outputs)
print (step_state)
print (step_next_inputs)
print (step_finished)

print (' --- output layer (Dense) --- ')
pprint(output_layer.__dict__)
print ('Output layer gets _trainable_variables')

# GRAPH
print (' --- Run Graph --- ')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run({
        'batch_size': decoder.batch_size,
        "first_finished": first_finished,
        "first_inputs": first_inputs,
        "first_state": first_state,
        "step_outputs": step_outputs,
        "step_state": step_state,
        "step_next_inputs": step_next_inputs,
        "step_finished": step_finished})
    print (' --- Show Results --- ')
    pprint (res)