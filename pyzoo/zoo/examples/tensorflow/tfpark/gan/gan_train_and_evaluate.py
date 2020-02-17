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
from bigdl.optim.optimizer import MaxIteration
from zoo.tfpark.gan.gan_estimator import GANEstimator

from zoo import init_nncontext
from zoo.tfpark import TFDataset, ZooOptimizer
from bigdl.dataset import mnist
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_gan.examples.mnist.networks import *
from tensorflow_gan.python.losses.losses_impl import *

MODEL_DIR = "/tmp/gan_model"
NOISE_DIM = 64


def get_data_rdd(dataset, sc):
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
    image_rdd = sc.parallelize(images_data)
    labels_rdd = sc.parallelize(labels_data)
    rdd = image_rdd.zip(labels_rdd) \
        .map(lambda rec_tuple: (((rec_tuple[0] / 255) - 0.5) * 2, np.array(rec_tuple[1])))
    return rdd


def eval():

    with tf.Graph().as_default() as g:
        noise = tf.random.normal(mean=0.0, stddev=1.0, shape=(50, NOISE_DIM))
        step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Generator"):
            one_hot = tf.one_hot(tf.concat([tf.range(0, 10)] * 5, axis=0), 10)
            fake_img = conditional_generator((noise, one_hot), is_training=False)
            fake_img = (fake_img * 128.0) + 128.0
            fake_img = tf.cast(fake_img, tf.uint8)
            tiled = tfgan.eval.image_grid(fake_img, grid_shape=(5, 10),
                                          image_shape=(28, 28),
                                          num_channels=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(MODEL_DIR)
            saver.restore(sess, ckpt)
            outputs, step_value = sess.run([tiled, step])
            plt.imsave("./image_{}.png".format(step_value), np.squeeze(outputs), cmap="gray")


if __name__ == "__main__":
    sc = init_nncontext()
    training_rdd = get_data_rdd("train", sc)

    def input_fn():
        dataset = TFDataset.from_rdd(training_rdd,
                                     features=(tf.float32, (28, 28, 1)),
                                     labels=(tf.int32, ()),
                                     batch_size=36)
        
        def map_func(tensors):
            images = tensors[0]
            labels = tensors[1]
            one_hot_label = tf.one_hot(labels, depth=10)
            in_graph_batch_size = tf.shape(images)[0]
            noise = tf.random.normal(mean=0.0, stddev=1.0, shape=(in_graph_batch_size, NOISE_DIM))
            generator_inputs = (noise, one_hot_label)
            discriminator_inputs = images
            return (generator_inputs, discriminator_inputs)

        dataset = dataset.map(map_func)
        return dataset

    opt = GANEstimator(
        generator_fn=conditional_generator,
        discriminator_fn=conditional_discriminator,
        generator_loss_fn=wasserstein_generator_loss,
        discriminator_loss_fn=wasserstein_discriminator_loss,
        generator_optimizer=ZooOptimizer(tf.train.AdamOptimizer(1e-5, 0.5)),
        discriminator_optimizer=ZooOptimizer(tf.train.AdamOptimizer(1e-4, 0.5)),
        model_dir=MODEL_DIR,
        session_config=tf.ConfigProto()
    )

    for i in range(20):
        opt.train(input_fn, MaxIteration(1000))
        eval()
