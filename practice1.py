import tensorflow as tf

# Build a graph
a = tf.constant(30)
b = tf.constant(20)

c = a*b

# launch the graph in session
sess = tf.Session()
File_Writer = tf.summary.FileWriter('C:/Users/shash/PycharmProjects/Tensorflow_models/graph', sess.graph)
print(sess.run(c))
sess.close()
# more boilerplates are required for good practice
