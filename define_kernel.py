'''
    https://www.tensorflow.org/tutorials/pdes
'''
import tensorflow as tf
import numpy as np

'''
    Kernel means local operation on 2D object (image, square lattice etc.)
'''

def make_kernel(arr):
    '''
        Take array as input
    '''
    arr = np.asarray(arr)
    arr = arr.reshape(list(arr.shape)+[1, 1])
    return tf.constant(arr, dtype=tf.float32)

def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, strides=[1,1,1,1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    """Compute the 2D laplacian of an array"""
    # https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                            [1.0, -6., 1.0],
                            [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


sess = tf.Session()

x = np.ones((12, 12), dtype=np.float32)
k = laplace(x)
k = sess.run(k)

print ('Laplacian')
print (k)
print (k.shape)

