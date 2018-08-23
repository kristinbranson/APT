## testing graph editing
import  tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import PoseTools

tf.reset_default_graph()
a = tf.placeholder(tf.float32,[3,3])
b = a + 3
c = b + 5
x = c+11
y = x+18

l = tf.zeros_like(b)
e1 = tf.identity(l)
d = tf.placeholder(tf.bool)
e = tf.cond(d,lambda: tf.identity(a), lambda: tf.identity(e1))
e2 = tf.identity(e)

gr = tf.get_default_graph()
pp = gr.get_operation_by_name('add')
ss = ge.sgv(pp.outputs[0])

pp1 = gr.get_operation_by_name('Identity')
ss1 = ge.sgv(pp1.outputs[0])
pp2 = gr.get_operation_by_name('Identity_1')
ss2 = ge.sgv(pp2.inputs[0])
# ge.detach_inputs(c)
# x = ge.graph_replace(x,{b:e})
# ge.detach_outputs(c)
m1 = ge.connect(ss2,ss)
m2 = ge.connect(ss,ss1)
# m = ge.swap_inputs(ss,ss1)
sess = tf.InteractiveSession()
PoseTools.output_graph('ge')
sess.close()

##
with tf.Session() as sess:
    q = sess.run(x,{a:np.zeros([3,3]),d:True})
    print q
    q = sess.run(x,{a:np.zeros([3,3]),d:False})
    print q
