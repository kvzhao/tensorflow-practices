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

print (x.shape)
print (x.get_shape())