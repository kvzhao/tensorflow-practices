import tensorflow as tf

sess = tf.Session()

'''
    get_variable
        name, shape
        trainable
    
        scope?
'''

x  = tf.get_variable('x', [5, 12, 13], 
    initializer=tf.random_normal_initializer())

# get the Varaible type

y = tf.Variable(initial_value=tf.random_normal([5,12,13], 0.0, 1.0), name='y')

sess.run(tf.global_variables_initializer())

print (x)
print (y)
#print (sess.run(x))
#print (sess.run(y))

print (x.name)
print (x.shape)
print (x.get_shape())

tf.reset_default_graph()

# build a graph
x = tf.placeholder(tf.float32, [None, 784])

with tf.variable_scope('simplest_network'):
    W = tf.get_variable('weight', shape=[784, 10], dtype=tf.float32)
    b = tf.get_variable('bias', shape=[10], dtype=tf.float32)
    y = tf.nn.softmax(tf.matmul(x, W) + b)


with tf.variable_scope('neural_newtork'):
    W1 = tf.get_variable('weight1', shape=[784, 256], dtype=tf.float32)
    b1 = tf.get_variable('bias1', shape=[256], dtype=tf.float32)
    W2 = tf.get_variable('weight2', shape=[256, 10], dtype=tf.float32)
    b2 = tf.get_variable('bias2', shape=[10], dtype=tf.float32)
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
    output = tf.nn.softmax(tf.add(tf.matmul(layer1, W2), b2))

def print_newtork(name):
    print ('{}:'.format(name))
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name):
        print ('\t{}\t{}'.format(var.name, var.get_shape()))

print_newtork('simplest_network')
'''
simplest_network:
        simplest_network/weight:0       (784, 10)
        simplest_network/bias:0 (10,)
'''

print_newtork('neural_newtork')
'''
neural_newtork:
        neural_newtork/weight1:0        (784, 256)
        neural_newtork/bias1:0  (256,)
        neural_newtork/weight2:0        (256, 10)
        neural_newtork/bias2:0  (10,)
'''
