import ray
import tensorflow as tf

from zoo.ray.allreduce.sgd import DistributedEstimator
from zoo.ray.data.dataset import RayDataSet


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

# session can not tbe serialize, so we cannot initialize it here.
# input_fn should return x and y which is ndarray or list of ndarray for multiple inputs

batch_size = 128

def input_fn():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    y_train = y_train.reshape((-1, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_test = y_test.reshape((-1, 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

def test_input_fn():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    y_train = y_train.reshape((-1, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_test = y_test.reshape((-1, 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return tf.data.Dataset.from_tensor_slices((x_test, y_test))

num_worker = 2
resource={"trainer": num_worker, "ps": num_worker }

ray.init(local_mode=False, resources=resource)

estimator = DistributedEstimator(model_fn=model_fn,
                     ray_dataset_train=RayDataSet.from_input_fn(input_fn, batch_size=batch_size),
                     num_worker=num_worker).train(1000)

# slow if the batch is small
print("ACC: {}".format(estimator.ray_model_resolved.evaluate(ray_dataset=RayDataSet.from_input_fn(test_input_fn, repeat=False, batch_size=10000)
                   )))
# ACC: 0.9697999954223633

# TODO: you need to stop tensorflow gracefully at the end of the training rather than throw exception
ray.shutdown()
