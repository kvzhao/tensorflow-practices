import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(5.0)

print (node1, node2)

sess = tf.Session()

print (sess.run([node1, node2]))

node3 = tf.add(node1, node2)

print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

print ("direct add: ", node1 + node2)
print ("sess.run(direct add)", sess.run(node1+node2))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print (sess.run(adder_node, {a:3.0, b:4.5}))
print (sess.run(adder_node, {a:[1.0, 2.0, 3.0], b:[4.0, 5.0, 6.0]}))

print ('linear model')
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.5], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x + b
