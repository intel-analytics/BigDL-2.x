import tensorflow as tf

from zoo.pipeline.api.net.tf_optimizer import GanOptimMethod

ds = tf.contrib.distributions
layers = tf.contrib.layers
tfgan = tf.contrib.gan
from tensorflow.contrib.gan.python.losses.python.losses_impl import wasserstein_discriminator_loss, wasserstein_generator_loss

def _generator_helper(
    noise, is_conditional, one_hot_labels, weight_decay, is_training):
  """Core MNIST generator.
  This function is reused between the different GAN modes (unconditional,
  conditional, etc).
  Args:
    noise: A 2D Tensor of shape [batch size, noise dim].
    is_conditional: Whether to condition on labels.
    one_hot_labels: Optional labels for conditioning.
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.
  Returns:
    A generated image in the range [-1, 1].
  """
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.contrib.framework.arg_scope(
        [layers.batch_norm], is_training=is_training):
      net = layers.fully_connected(noise, 1024)
      if is_conditional:
        net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
      net = layers.fully_connected(net, 7 * 7 * 128)
      net = tf.reshape(net, [-1, 7, 7, 128])
      net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
      net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
      # Make sure that generator output is in the same range as `inputs`
      # ie [-1, 1].
      net = layers.conv2d(
          net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

      return net


def unconditional_generator(noise, weight_decay=2.5e-5, is_training=True):
  """Generator to produce unconditional MNIST images.
  Args:
    noise: A single Tensor representing noise.
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.
  Returns:
    A generated image in the range [-1, 1].
  """
  return _generator_helper(noise, False, None, weight_decay, is_training)

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)


def _discriminator_helper(img, is_conditional, one_hot_labels, weight_decay):
  """Core MNIST discriminator.
  This function is reused between the different GAN modes (unconditional,
  conditional, etc).
  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    is_conditional: Whether to condition on labels.
    one_hot_labels: Labels to optionally condition the network on.
    weight_decay: The L2 weight decay.
  Returns:
    Final fully connected discriminator layer. [batch_size, 1024].
  """
  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.fully_connected],
      activation_fn=_leaky_relu, normalizer_fn=None,
      weights_regularizer=layers.l2_regularizer(weight_decay),
      biases_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.conv2d(img, 64, [4, 4], stride=2)
    net = layers.conv2d(net, 128, [4, 4], stride=2)
    net = layers.flatten(net)
    if is_conditional:
      net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
    net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

    return net


def unconditional_discriminator(img, weight_decay=2.5e-5):
  """Discriminator network on unconditional MNIST digits.
  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    weight_decay: The L2 weight decay.
  Returns:
    Logits for the probability that the image is real.
  """
  net = _discriminator_helper(img, False, None, weight_decay)
  return layers.linear(net, 1)

import tensorflow as tf
from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import numpy as np
import sys

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *

sc = init_nncontext()

def get_data_rdd(dataset):
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
    image_rdd = sc.parallelize(images_data)
    labels_rdd = sc.parallelize(labels_data)
    rdd = image_rdd.zip(labels_rdd) \
        .map(lambda rec_tuple: [((rec_tuple[0] / 255) - 0.5) * 2,
                                np.array(rec_tuple[1])])
    return rdd


training_rdd = get_data_rdd("train")
testing_rdd = get_data_rdd("test")
dataset = TFDataset.from_rdd(training_rdd,
                             names=["features", "labels"],
                             shapes=[[28, 28, 1], []],
                             types=[tf.float32, tf.int32],
                             batch_size=32)

real_images, dummy_labels = dataset.tensors

counter = tf.Variable(0, dtype=tf.int32)

batch_size = tf.shape(real_images)[0]

noise = tf.random.normal(mean=0.0, stddev=1.0, shape=(batch_size, 10))

is_discriminator_phase = tf.equal(tf.mod(counter, 2), 0)

with tf.control_dependencies([is_discriminator_phase]):
    increase_counter = tf.assign_add(counter, 1)

with tf.variable_scope("generator"):
    fake_img = unconditional_generator(noise=noise)

images = tf.cond(is_discriminator_phase, lambda: tf.concat([fake_img, real_images], axis=0), lambda: fake_img)

with tf.variable_scope("discriminator"):
    fake_logits = unconditional_discriminator(fake_img)

with tf.variable_scope("discriminator", reuse=True):
    real_logits = unconditional_discriminator(real_images)

generator_loss = wasserstein_generator_loss(fake_logits)
generator_variables = tf.trainable_variables("generator")

generator_grads = tf.gradients(generator_loss, generator_variables)

discriminator_loss = wasserstein_discriminator_loss(real_logits, fake_logits)
discriminator_variables = tf.trainable_variables("discriminator")
discriminator_grads = tf.gradients(discriminator_loss, discriminator_variables)


variables = generator_variables + discriminator_variables
g_grads = tf.cond(is_discriminator_phase, lambda: [tf.zeros_like(grad) for grad in generator_grads], lambda: generator_grads)
d_grads = tf.cond(is_discriminator_phase, lambda: discriminator_grads, lambda: [tf.zeros_like(grad) for grad in discriminator_grads])
loss = tf.cond(is_discriminator_phase, lambda: discriminator_loss, lambda: generator_loss)

grads = g_grads + d_grads

g_param_size = sum([np.product(g.shape) for g in g_grads])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optimizer = TFOptimizer(loss, GanOptimMethod(Adam(1e-4, beta1=0.5), Adam(1e-3, beta1=0.5), g_param_size.value), sess=sess,
                            dataset=dataset, inputs=[real_images, dummy_labels],
                 grads=grads, variables=variables, graph=tf.get_default_graph(), updates=[increase_counter])

    optimizer.optimize(MaxIteration(5000))
    saver = tf.train.Saver()
    saver.save(optimizer.sess, "/tmp/gan/model-5000")


