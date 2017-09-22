import numpy as np
import tensorflow as tf

from tensorflow.python.ops.script_ops import py_func

def get_neighbor(site, L):
    pbc = lambda s, d, l: ((s+d)%l + l) % l
    x, y = int(site%L), int(site/L)
    neighbors = []
    xp = pbc(x, +1, L)
    xm = pbc(x, -1, L)
    yp = pbc(y, +1, L)
    ym = pbc(y, -1, L)
    neighbors.append(xp + y  * L)
    neighbors.append(x  + ym * L)
    neighbors.append(xm + y  * L)
    neighbors.append(x  + yp * L)
    if (x+y) % 2 == 0:
        # even
        neighbors.append(xp + yp  * L)
        neighbors.append(xm + ym  * L)
    else:
        # odd
        neighbors.append(xm + yp  * L)
        neighbors.append(xp + ym  * L)
    return neighbors

def cal_energy(state, L=32):
    eng = 0.0
    J = 1.0
    for site, spin in enumerate(state):
        neighbors = get_neighbor(site, L)
        se = np.sum(state[neighbors], dtype=np.float32)
        eng += J * spin * se
    eng = eng / (2.0*L**2)
    return np.float32(eng)

s = np.load('ice.npy')
s = s.astype(np.float32)
L = 32

# Compute energy using numpy
print (s)
print ('Compute ice state energy by numpy: {}'.format(cal_energy(s, L)))

# compute energy using tensorflow
inp = tf.placeholder(tf.float32, shape=[None,], name='inplaceholder')

tf_cal_energy = tf.py_func(cal_energy, [inp], tf.float32)

print (tf_cal_energy)

sess = tf.Session()
print ('Compute ice state energy from tensorflow: {}'.format(sess.run(tf_cal_energy, feed_dict={inp: s})))