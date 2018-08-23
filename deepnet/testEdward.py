from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from edward.models import Categorical, Mixture, Normal
from tensorflow.contrib import slim
from scipy import stats
from sklearn.model_selection import train_test_split

def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
  """Plots the mixture of Normal models to axis=ax comp=True plots all
  components of mixture model
  """
  x = np.linspace(-10.5, 10.5, 250)
  final = np.zeros_like(x)
  for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
    temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
    final = final + temp
    if comp:
      ax.plot(x, temp, label='Normal ' + str(i))
  ax.plot(x, final, label='Mixture of Normals ' + label)
  ax.legend(fontsize=13)


def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
  """Draws samples from mixture model.

  Returns 2 d array with input X and sample from prediction of mixture model.
  """
  samples = np.zeros((amount, 2))
  n_mix = len(pred_weights[0])
  to_choose_from = np.arange(n_mix)
  for j, (weights, means, std_devs) in enumerate(
          zip(pred_weights, pred_means, pred_std)):
    index = np.random.choice(to_choose_from, p=weights)
    samples[j, 1] = np.random.normal(means[index], std_devs[index], size=1)
    samples[j, 0] = x[j]
    if j == amount - 1:
      break
  return samples

def build_toy_dataset(N):
  y_data = np.random.uniform(-10.5, 10.5, N)
  r_data = np.random.normal(size=N)  # random noise
  x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
  x_data = x_data.reshape((N, 1))
  return train_test_split(x_data, y_data, random_state=42)


ed.set_seed(24)

N = 50000  # number of data points
D = 1  # number of features
K = 20  # number of mixture components

X_train, X_test, y_train, y_test = build_toy_dataset(N)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))
# sns.regplot(X_train, y_train, fit_reg=False)
# plt.show()

X_ph = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])


##
def neural_network(X):
  """loc, scale, logits = NN(x; theta)"""
  # 2 hidden layers with 15 hidden units
  hidden1 = slim.fully_connected(X, 60,normalizer_fn=slim.batch_norm)
  hidden2 = slim.fully_connected(hidden1, 60,normalizer_fn=slim.batch_norm)
  locs = slim.fully_connected(hidden2, K, activation_fn=None)
  o_scales = slim.fully_connected(hidden2, K, activation_fn=tf.exp)
  scales = tf.minimum(20.,tf.maximum(0.0001,o_scales))
  logits = slim.fully_connected(hidden2, K, activation_fn=None)
  return locs, scales, logits, hidden1

def myNormal(loc,scale):
  nn = Normal(loc,scale)
  tt = Normal(loc,10.)
  return Mixture(Categorical(probs=[[0.95,0.05]]),components=[nn,tt])

locs, scales, logits,hidden1 = neural_network(X_ph)
cat = Categorical(logits=logits)
components = [Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(locs)),
                     tf.unstack(tf.transpose(scales)))]
y = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))
# Note: A bug exists in Mixture which prevents samples from it to have
# a shape of [None]. For now fix it using the value argument, as
# sampling is not necessary for MAP estimation anyways.

#

# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.
n_epoch = 1000

inference = ed.MAP(data={y: y_ph})
inference.initialize(var_list=tf.trainable_variables(),n_iter=n_epoch)

sess = ed.get_session()
tf.global_variables_initializer().run()

train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
  info_dict = inference.update(feed_dict={X_ph: X_train, y_ph: y_train})
  train_loss[i] = info_dict['loss']
  test_loss[i] = sess.run(inference.loss,
                          feed_dict={X_ph: X_test, y_ph: y_test})
  inference.print_progress(info_dict)


pred_weights, pred_means, pred_std = \
    sess.run([tf.nn.softmax(logits), locs, scales], feed_dict={X_ph: X_test})

obj = [0, 4, 6]
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 6))

plot_normal_mix(pred_weights[obj][0], pred_means[obj][0],
                pred_std[obj][0], axes[0], comp=False)
axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)

plot_normal_mix(pred_weights[obj][2], pred_means[obj][2],
                pred_std[obj][2], axes[1], comp=False)
axes[1].axvline(x=y_test[obj][2], color='black', alpha=0.5)

plot_normal_mix(pred_weights[obj][1], pred_means[obj][1],
                pred_std[obj][1], axes[2], comp=False)
axes[2].axvline(x=y_test[obj][1], color='black', alpha=0.5)
plt.show()


a = sample_from_mixture(X_test, pred_weights, pred_means,
                        pred_std, amount=len(X_test))
sns.jointplot(a[:, 0], a[:, 1], kind="hex", color="#4CB391",
              ylim=(-10, 10), xlim=(-14, 14))
plt.show()
