import tensorflow as tf

# Build a graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adding_node = a+b

# launch the graph in session
sess = tf.Session()
print(sess.run(adding_node, {a:[1,3], b:[2,4]}))