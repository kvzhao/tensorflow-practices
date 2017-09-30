import tensorflow as tf

g1 = tf.get_default_graph()
print (g1)

g2 = tf.Graph()
print (g2)


# set graph two as default graph
with g2.as_default():
    print (g2 == tf.get_default_graph())