import tensorflow as tf
sess = tf.Session()
devices = sess.list_devices()

for d in devices:
    print (d.name)

sess.close()