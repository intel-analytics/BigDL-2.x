#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorflow as tf

from tensorflow_gan.python.losses.losses_impl import wasserstein_discriminator_loss, \
    wasserstein_generator_loss

import tensorflow_gan as tfgan

ds = tf.contrib.distributions
layers = tf.contrib.layers

def discriminator_loss_fn(real_outputs, gen_outputs):
    return wasserstein_discriminator_loss(real_outputs, gen_outputs)


def generator_loss_fn(gen_outputs):
    return wasserstein_generator_loss(gen_outputs)


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


