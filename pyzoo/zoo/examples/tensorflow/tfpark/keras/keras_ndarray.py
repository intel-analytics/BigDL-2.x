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
import sys

import tensorflow as tf
# from zoo import init_nncontext
from bigdl.dataset import mnist
# from zoo.tfpark import KerasModel
boundary_epochs=[30, 60, 80, 90]
decay_rates=[1, 0.1, 0.01, 0.001, 1e-4]
batches_per_epoch = 312
initial_learning_rate = 0.1 * 4096 // 256
boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
vals = [initial_learning_rate * decay for decay in decay_rates]

def learning_rate_fn():
    import tensorflow as tf
    global_step = tf.compat.v1.train.get_or_create_global_step()
    print(global_step)
    lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)
    warmup_steps = int(batches_per_epoch * 5)
    warmup_lr = (
        initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
        warmup_steps, tf.float32))
    return tf.cond(pred=global_step < warmup_steps, true_fn=lambda: warmup_lr, false_fn=lambda: lr)

def compile_args_creator(config):
    momentum = config["momentum"]
    import tensorflow.keras as keras

    opt = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=learning_rate_fn(),
        momentum=momentum
    )
    param = dict(loss=keras.losses.sparse_categorical_crossentropy, optimizer=opt,
                 metrics=['accuracy', 'top_k_categorical_accuracy'])
    return param


def main(max_epoch):
    # _ = init_nncontext()

    (training_images_data, training_labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    (testing_images_data, testing_labels_data) = mnist.read_data_sets("/tmp/mnist", "test")

    training_images_data = (training_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    testing_images_data = (testing_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD

    model = tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(64, activation='relu'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )
    import tensorflow.keras as keras
    keras.backend.get_session().run(tf.global_variables_initializer())
    opt = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=learning_rate_fn(),
        momentum=0.9
    )
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    class SGDLearningRateTracker(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            optimizer = self.model.optimizer
            lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
            print('\nLR: {:.6f}\n'.format(lr))

    model.fit(training_images_data, training_labels_data, callbacks=[SGDLearningRateTracker()])

    # keras_model = KerasModel(model)
    #
    # keras_model.fit(training_images_data,
    #                 training_labels_data,
    #                 validation_data=(testing_images_data, testing_labels_data),
    #                 epochs=max_epoch,
    #                 batch_size=320,
    #                 distributed=True)
    #
    # result = keras_model.evaluate(testing_images_data, testing_labels_data,
    #                               distributed=True, batch_per_thread=80)

    # print(result)
    # # >> [0.08865142822265625, 0.9722]
    #
    # # the following assert is used for internal testing
    # assert result['acc Top1Accuracy'] > 0.95
    #
    # keras_model.save_weights("/tmp/mnist_keras.h5")


if __name__ == '__main__':

    max_epoch = 5

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
    main(max_epoch)
