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
import tempfile
import os
import tensorflow as tf
import numpy as np

from zoo.pipeline.api.net import TFOptimizer
from zoo.pipeline.api.net.tf_optimizer import GanOptimMethod


class GANOptimizer(object):

    def __init__(self,
                 generator_fn,
                 discriminator_fn,
                 generator_loss_fn,
                 discriminator_loss_fn,
                 generator_optim_method,
                 discriminator_optim_method,
                 dataset,
                 noise_generator,
                 generator_steps=1,
                 discriminator_steps=1,
                 checkpoint_path=None,
                 ):
        self._generator_fn = generator_fn
        self._discriminator_fn = discriminator_fn
        self._generator_loss_fn = generator_loss_fn
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_steps = generator_steps
        self._discriminator_steps = discriminator_steps
        self._generator_optim_method = generator_optim_method
        self._discriminator_optim_method = discriminator_optim_method
        self._dataset = dataset
        self._noise_generator = noise_generator

        if checkpoint_path is None:
            folder = tempfile.mkdtemp()
            self.checkpoint_path = os.path.join(folder, "gan_model")
        else:
            self.checkpoint_path = checkpoint_path

    def optimize(self, end_trigger):

        with tf.Graph().as_default() as g:

            real_images = self._dataset.tensors[0]
            counter = tf.Variable(0, dtype=tf.int32)

            batch_size = tf.shape(real_images)[0]

            noise = self._noise_generator(batch_size)

            is_discriminator_phase = tf.equal(tf.mod(counter, 2), 0)

            with tf.control_dependencies([is_discriminator_phase]):
                increase_counter = tf.assign_add(counter, 1)

            with tf.variable_scope("generator"):
                fake_img = self._generator_fn(noise)

            with tf.variable_scope("discriminator"):
                fake_logits = self._discriminator_fn(fake_img)

            with tf.variable_scope("discriminator", reuse=True):
                real_logits = self._discriminator_fn(real_images)

            with tf.name_scope("generator_loss"):
                generator_loss = self._generator_loss_fn(fake_logits)

            with tf.name_scope("discriminator_loss"):
                discriminator_loss = self._discriminator_loss_fn(real_logits, fake_logits)

            generator_variables = tf.trainable_variables("generator")
            generator_grads = tf.gradients(generator_loss, generator_variables)
            discriminator_variables = tf.trainable_variables("discriminator")
            discriminator_grads = tf.gradients(discriminator_loss, discriminator_variables)

            variables = generator_variables + discriminator_variables
            g_grads = tf.cond(is_discriminator_phase, lambda: [tf.zeros_like(grad) for grad in generator_grads],
                              lambda: generator_grads)
            d_grads = tf.cond(is_discriminator_phase, lambda: discriminator_grads,
                              lambda: [tf.zeros_like(grad) for grad in discriminator_grads])
            loss = tf.cond(is_discriminator_phase, lambda: discriminator_loss, lambda: generator_loss)

            grads = g_grads + d_grads

            g_param_size = sum([np.product(g.shape) for g in g_grads])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                optimizer = TFOptimizer(loss, GanOptimMethod(self._discriminator_loss_fn, self._generator_optim_method,
                                                             g_param_size.value), sess=sess,
                                        dataset=self._dataset, inputs=self._dataset.tensors,
                                        grads=grads, variables=variables, graph=g,
                                        updates=[increase_counter])
                optimizer.optimize(end_trigger)
                steps = sess.run(counter)
                saver = tf.train.Saver()
                saver.save(optimizer.sess, self.checkpoint_path, global_step=steps)





