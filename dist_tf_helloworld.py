import tensorflow as tf

c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()

with tf.Session(server.target) as sess:
    print( sess.run(c) )
