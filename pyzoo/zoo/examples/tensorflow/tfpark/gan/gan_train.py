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

from zoo.tfpark.gan.gan_estimator import GANEstimator

from zoo import init_nncontext
from zoo.tfpark import TFDataset
from bigdl.optim.optimizer import *
import numpy as np

from bigdl.dataset import mnist
from zoo.examples.tensorflow.tfpark.gan.gan_model import *


def get_data_rdd(dataset):
    (images_data, labels_data) = mnist.read_data_sets("/tmp/mnist", dataset)
    image_rdd = sc.parallelize(images_data).map(lambda img: [((img / 255) - 0.5) * 2])
    return image_rdd


if __name__ == "__main__":
    sc = init_nncontext()
    training_rdd = get_data_rdd("train")
    dataset = TFDataset.from_rdd(training_rdd,
                                 features=(tf.float32, (28, 28, 1)),
                                 batch_size=32)

    opt = GANEstimator(
        generator_fn=lambda noise: unconditional_generator(noise),
        discriminator_fn=lambda real_data: unconditional_discriminator(real_data),
        generator_loss_fn=generator_loss_fn,
        discriminator_loss_fn=discriminator_loss_fn,
        generator_optim_method=Adam(1e-3, beta1=0.5),
        discriminator_optim_method=Adam(1e-4, beta1=0.5),
        noise_generator=lambda batch_size: tf.random.normal(mean=0.0, stddev=1.0, shape=(batch_size, 10)),
        checkpoint_path="/tmp/gan_model/model"
    )

    opt.train(dataset, MaxIteration(5000))
