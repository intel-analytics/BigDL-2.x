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
from unittest import TestCase
from zoo.orca.learn.tf.tf_ray_estimator import TFRayEstimator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400


def create_config(batch_size):
    import tensorflow as tf

    return {
        # todo: batch size needs to scale with # of workers
        "batch_size": batch_size,
        "fit_config": {
            "epochs": 5,
            "steps_per_epoch": NUM_TRAIN_SAMPLES // batch_size,
            "callbacks": [tf.keras.callbacks.ModelCheckpoint(
                            "/tmp/checkpoint/keras_ckpt", monitor='val_loss', verbose=0, save_best_only=False,
                            save_weights_only=False, mode='auto', save_freq='epoch')]
        },
        "evaluate_config": {
            "steps": NUM_TEST_SAMPLES // batch_size,
        },
        "compile_config": {
            "optimizer": "sgd",
            "loss": "mean_squared_error",
            "metrics": ["mean_squared_error"]
        }
    }


def linear_dataset(a=2, size=1000):
    x = np.random.rand(size)
    y = x / 2

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    return x, y


def simple_dataset(config):
    batch_size = config["batch_size"]
    x_train, y_train = linear_dataset(size=NUM_TRAIN_SAMPLES)
    x_test, y_test = linear_dataset(size=NUM_TEST_SAMPLES)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).repeat().batch(
        batch_size)
    test_dataset = test_dataset.repeat().batch(batch_size)

    return train_dataset, test_dataset


def simple_model(config):
    model = Sequential([Dense(10, input_shape=(1, )), Dense(1)])
    return model


def compile_args():
    import tensorflow as tf
    args = {
        "optimizer": tf.keras.optimizers.Adam(),
        "loss": "mean_squared_error",
        "metrics": ["mean_squared_error"]
    }
    return args


class TestTFRayEstimator(TestCase):

    def test_fit_and_evaluate(self):

        trainer = TFRayEstimator(
            model_creator=simple_model,
            compile_args=compile_args(),
            data_creator=simple_dataset,
            verbose=True,
            config=create_config(32))

        # model baseline performance
        start_stats = trainer.evaluate()
        print(start_stats)

        # train for 2 epochs
        trainer.fit()
        trainer.fit()

        # model performance after training (should improve)
        end_stats = trainer.evaluate()
        print(end_stats)

        # sanity check that training worked
        dloss = end_stats["validation_loss"] - start_stats["validation_loss"]
        dmse = (end_stats["validation_mean_squared_error"] -
                start_stats["validation_mean_squared_error"])
        print(f"dLoss: {dloss}, dMSE: {dmse}")

        assert dloss < 0 and dmse < 0, "training sanity check failed. loss increased!"
