import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np

from zoo.pipeline.api.net import TFDataset
from zoo.tfpark import estimator
from zoo.tfpark.estimator import EstimatorSpec, Estimator
from zoo.tfpark.model import Model


class TestTFParkEstimator(ZooTestCase):

    def create_model_fn(self):
        def cnn_model_fn(features, labels, mode):
            """Model function for CNN."""
            # Input Layer
            input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=10)

            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # Calculate Loss (for both TRAIN and EVAL modes)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                # train_op = optimizer.minimize(
                #     loss=loss,
                #     global_step=tf.train.get_global_step())
                return EstimatorSpec(mode=mode, loss=loss, optimizer=optimizer)

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])
            }
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return cnn_model_fn

    def create_training_data(self):
        ((train_data, train_labels),
         (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

        train_data = train_data / np.float32(255)
        train_labels = train_labels.astype(np.int32)  # not required

        eval_data = eval_data / np.float32(255)
        eval_labels = eval_labels.astype(np.int32)  # not required
        return train_data, train_labels, eval_data, eval_labels

    def create_training_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y).map(lambda x: list(x))

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features", "labels"],
                                     shapes=[[10], []],
                                     types=[tf.float32, tf.int32],
                                     batch_size=4,
                                     val_rdd=rdd
                                     )
        return dataset

    def create_evaluation_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y).map(lambda x: list(x))

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features", "labels"],
                                     shapes=[[10], []],
                                     types=[tf.float32, tf.int32],
                                     batch_per_thread=1
                                     )
        return dataset

    def create_predict_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)

        rdd = self.sc.parallelize(x)

        rdd = rdd.map(lambda x: [x])

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features"],
                                     shapes=[[10]],
                                     types=[tf.float32],
                                     batch_per_thread=1
                                     )
        return dataset

    def test_training_with_ndarray(self):

        model_fn = self.create_model_fn()

        estimator = Estimator.from_model_fn(model_fn)

        train_data, train_labels, eval_data, eval_labels = self.create_training_data()

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

        estimator.train(train_input_fn, steps=1000)



if __name__ == "__main__":
    # pytest.main([__file__])
    test = TestTFParkEstimator()
    test.test_training_with_ndarray()
