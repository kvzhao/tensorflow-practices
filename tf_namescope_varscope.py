"""Exploer difference between 
    tf.name_scope() and tf.variable_scope()

    Reference: https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
"""


import tensorflow as tf

with tf.name_scope("name_scope"):
    v1 = tf.get_variable(name="var1", shape=[1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    m = tf.multiply(v1, v2)
    v3 = tf.get_variable(name="name_scope/var3", shape=[2], dtype=tf.float32)

print (v1.name) # <tf.Variable 'var1:0' shape=(1,) dtype=float32_ref>
# Not in the same scope.

print (v2.name) # <tf.Variable 'name_scope/var2:0' shape=() dtype=float32_ref>
print (m.name) # Tensor("name_scope/Mul:0", shape=(1,), dtype=float32)
print (v3.name) # <tf.Variable 'name_scope/var3:0' shape=(2,) dtype=float32_ref>
#  this is the method of assigning the name scope (by name)

# Compare with varaible scope
tf.reset_default_graph()
print ("Graph Reset.")

with tf.variable_scope("var_scope"):
    v1 = tf.get_variable(name="var1", shape=[1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    m = tf.multiply(v1, v2)
    v3 = tf.get_variable(name="name_scope/var3", shape=[2], dtype=tf.float32)
    v4 = tf.get_variable(name="var_scope/var4", shape=[2], dtype=tf.float32)

print (v1.name) # <tf.Variable 'var_scope/var1:0' shape=(1,) dtype=float32_ref>
print (v2.name) # <tf.Variable 'var_scope/var2:0' shape=() dtype=float32_ref>
# Share the scope!
print (m.name) # Tensor("var_scope/Mul:0", shape=(1,), dtype=float32)

print (v3.name) # <tf.Variable 'var_scope/name_scope/var3:0' shape=(2,) dtype=float32_ref>
print (v4.name) # <tf.Variable 'var_scope/var_scope/var4:0' shape=(2,) dtype=float32_ref>
# Now, the scope can not be distinguished by assigning name

tf.reset_default_graph()
print ("Graph Reset.")

"""
    variable scope can share var by using get_variable()
"""

with tf.name_scope("name_scope_A"):
    with tf.variable_scope("var_scope"):
        v = tf.get_variable("var", shape=[10], dtype=tf.float32)
        w = tf.Variable(1, name="w", dtype=tf.float32)

with tf.name_scope("name_scope_Z"):
    with tf.variable_scope("var_scope", reuse=True):
        # Reuse if exist
        vz = tf.get_variable("var", shape=[10], dtype=tf.float32)
        wz = tf.Variable(1, name="w", dtype=tf.float32)


assert v == vz
# Share the same memory and name
print (v.name)
print (vz.name)

assert w != wz
# Differ by name scope
print (w.name)
print (wz.name)