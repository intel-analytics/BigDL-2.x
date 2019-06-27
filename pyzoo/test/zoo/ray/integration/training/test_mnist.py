import ray
import tensorflow as tf

from zoo.ray.allreduce.model import RayModel
from zoo.ray.data.dataset import RayDataSet
import numpy as np


def model_fn():
    """
    You should add your definition here and then return (input, output, target, loss, optimizer)
    :return:
    """
    images = tf.keras.layers.Input((28, 28, 1))
    target = tf.keras.layers.Input((1, ))
    input1 = tf.keras.layers.Flatten(input_shape=(28, 28, 1))(images)
    dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(input1)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, dense2)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    return images, dense2, target, loss, optimizer

num_worker = 2
resource={"trainer": num_worker, "ps": num_worker }
ray.init(local_mode=False, resources=resource)
batch_size = 128

def input_fn():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train) = np.random.uniform(0, 1, size=(600, 28, 28, 1)), np.random.uniform(0, 1, size=(600, 1))
    x_train = x_train.reshape((-1, 28, 28, 1))
    y_train = y_train.reshape((-1, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_test = y_test.reshape((-1, 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    return dataset

def test_input_fn():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    y_train = y_train.reshape((-1, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_test = y_test.reshape((-1, 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return tf.data.Dataset.from_tensor_slices((x_test, y_test))



rayModel = RayModel.from_model_fn(model_fn)
rayModel.fit(ray_dataset_train=RayDataSet.from_input_fn(input_fn, batch_size=batch_size),
            num_worker=num_worker,
            steps=1000)

# slow if the batch is small
print("ACC: {}".format(rayModel.resolved_model.evaluate(ray_dataset=RayDataSet.from_input_fn(test_input_fn, repeat=False, batch_size=10000)
                   )))
# ACC: 0.9697999954223633

ray.shutdown()
