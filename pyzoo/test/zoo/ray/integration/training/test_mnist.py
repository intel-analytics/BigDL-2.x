import ray
import tensorflow as tf

from zoo.ray.allreduce.model import RayModel
from zoo.ray.allreduce.sgd import DistributedEstimator
from zoo.ray.data.dataset import RayDataSet

# import pickle
# pickle.dumps(calc_accuracy)
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
# input_fn should return x and y which is ndarray or list of ndarray(multiple input)

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

# dataset = RayDataSet.from_input_fn(input_fn, batch_size=batch_size, repeat=False)
# while dataset.has_next():
#     data = dataset.next_batch()

# def calc_accuracy(sess, inputs_op, outputs_op, targets_op, input_data, output_data):
#     with tf.name_scope('accuracy'):
#         # label [-1, 1] not one-hot encoding. If the shape mismatch, the result would be incorrect
#         # as `tf.equal` would broadcast automatically during the comparing stage.
#         correct_prediction = tf.equal(tf.argmax(outputs_op[0], 1),
#                                       tf.cast(tf.reshape(targets_op[0], (-1,)), tf.int64))
#         correct_prediction = tf.cast(correct_prediction, tf.float32)
#         accuracy = tf.reduce_mean(correct_prediction)
#         return sess.run(accuracy,
#                         feed_dict={targets_op[0]: output_data, inputs_op[0]: input_data})
#
# input_op, output_op, label_op, loss, optimizer = model_fn()
#
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape((-1, 28, 28, 1))
# y_train = y_train.reshape((-1, 1))
# x_test = x_test.reshape((-1, 28, 28, 1))
# y_test = y_test.reshape((-1, 1))
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
#    intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2)) as sess:
#     train_op = optimizer.minimize(loss)
#     tf.global_variables_initializer().run()
#     dataset = RayDataSet.from_input_fn(input_fn, batch_size=batch_size, repeat=True)
#
#     for _ in range(1000):
#         if dataset.has_next():
#             x, y = dataset.next_batch()
#         sess.run(train_op, feed_dict={input_op: x, label_op: y})
#
#     # test_dataset = RayDataSet.from_input_fn(test_input_fn, repeat=False, batch_size=10000)
#     # if test_dataset.has_next():
#     #     test_input, test_label = test_dataset.next_batch()
#
#     print(calc_accuracy(sess, [input_op], [output_op], [label_op], x_test, y_test))
#
#



num_worker = 2
resource={"trainer": num_worker, "ps": num_worker }
#
# sc = init_spark_on_local(cores=44)
#
# ray_ctx = RayContext(sc=sc, local_ray_node_num=2)
# ray_ctx.init()
ray.init(local_mode=False, resources=resource)

estimator = DistributedEstimator(model_fn=model_fn,
                     ray_dataset_train=RayDataSet.from_input_fn(input_fn, batch_size=batch_size),
                     num_worker=num_worker).train(1000)
# rayModel = RayModel.from_model_fn(model_fn).resolve()

# very slow if the batch is small
print("ACC: {}".format(estimator.ray_model_resolved.evaluate(ray_dataset=RayDataSet.from_input_fn(test_input_fn, repeat=False, batch_size=10000)
                   )))

# estimator.evaluate(ray_dataset=RayDataSet.from_input_fn(input_fn, repeat=False, batch_size=batch_size),
#                    metric_fn=calc_accuracy)

# TODO: you need to stop tensorflow gracefully while end of the training rather than throw exception
ray.shutdown()
