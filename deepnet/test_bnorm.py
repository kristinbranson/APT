import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
#from batch_norm import batch_norm_new as batch_norm

gpu()
tf.reset_default_graph()

# Two layer network.
phase_train_mdn = tf.placeholder(tf.bool)

x_flat = tf.placeholder(tf.float32,[3,5])
weights1 = tf.get_variable("weights1", [5, 3],
                           initializer=tf.random_normal_initializer())
biases1 = tf.get_variable("biases1", 1,
                          initializer=tf.constant_initializer(0))
hidden1_b = tf.matmul(x_flat, weights1)
hidden1 = batch_norm(hidden1_b, decay=0.8,
                     is_training=phase_train_mdn)
hidden1 = hidden1 + biases1

weights2 = tf.get_variable("weights2", [3, 1],
                           initializer=tf.contrib.layers.xavier_initializer())
biases2 = tf.get_variable("biases2", 1,
                          initializer=tf.constant_initializer(0))
hidden2 = tf.matmul(hidden1, weights2) + biases2

y = tf.placeholder(tf.float32,[3,])
step = tf.placeholder(tf.int32)

loss = tf.nn.l2_loss(hidden2-y)
#

learning_rate = tf.train.exponential_decay(
    0.1, step, 100, 0.9)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.InteractiveSession()
x_in = np.random.rand(3, 5)
y_in = np.random.rand(3)
sess.run(tf.variables_initializer(tf.global_variables()),
         feed_dict={x_flat:x_in,y:y_in,step:0})
for ndx in range(100):
    x_in = np.random.rand(3,5)
    y_in = np.random.rand(3)
    sess.run(opt,feed_dict={x_flat:x_in,y:y_in,
                            step:0,phase_train_mdn:True})

x_in1 = np.random.rand(3,5)
x_in2 = x_in1.copy()
x_in2[1:,] = x_in1[0,:]

##
y_1,yy_1 = sess.run([hidden1,hidden1_b],
                    feed_dict={x_flat:x_in1,y:y_in,
                               step:0,phase_train_mdn:False})
y_2,yy_2 = sess.run([hidden1,hidden1_b],
                    feed_dict={x_flat:x_in2,y:y_in,
                               step:0,phase_train_mdn:False})
y_3,yy_3 = sess.run([hidden1,hidden1_b],
                    feed_dict={x_flat:x_in1,y:y_in,
                               step:0,phase_train_mdn:True})
y_4,yy_4 = sess.run([hidden1,hidden1_b],
                    feed_dict={x_flat:x_in1,y:y_in,
                               step:0,phase_train_mdn:True})
y_5,yy_5 = sess.run([hidden1,hidden1_b],
                    feed_dict={x_flat:x_in1,y:y_in,
                               step:0,phase_train_mdn:False})

print y_1
print y_2
print y_3
print y_4
print y_5
