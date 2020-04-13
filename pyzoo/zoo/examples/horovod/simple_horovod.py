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

import argparse
from zoo import init_spark_on_yarn
from zoo.horovod.horovod_ray_runner import HorovodRayRunner
from zoo.ray.util.raycontext import RayContext


def run_horovod():
    import tensorflow as tf
    import horovod.tensorflow.keras as hvd

    hvd.init()
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(10000).batch(128)

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    opt = tf.optimizers.Adam(0.001 * hvd.size())

    opt = hvd.DistributedOptimizer(opt)

    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
    ]

    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
    verbose = 1 if hvd.rank() == 0 else 0
    result = mnist_model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=5, verbose=verbose)
    return result

parser = argparse.ArgumentParser()
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                         " configuration folder. Otherwise, turn on local mode.")
parser.add_argument("--slave_num", type=int, default=2,
                    help="The number of slave nodes")
parser.add_argument("--conda_name", type=str,
                    help="The name of conda environment.")
parser.add_argument("--iface", type=str, default="eth0")
parser.add_argument("--executor_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--executor_memory", type=str, default="10g",
                    help="The size of slave(executor)'s memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_memory", type=str, default="2g",
                    help="The size of driver's memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--extra_executor_memory_for_ray", type=str, default="20g",
                    help="The extra executor memory to store some data."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--object_store_memory", type=str, default="4g",
                    help="The memory to store data on local."
                         "You can change it depending on your own cluster setting.")

if __name__ == "__main__":
    args = parser.parse_args()
    sc = init_spark_on_yarn(
        hadoop_conf=args.hadoop_conf,
        conda_name=args.conda_name,
        num_executor=args.slave_num,
        executor_cores=args.executor_cores,
        executor_memory=args.executor_memory,
        driver_memory=args.driver_memory,
        driver_cores=args.driver_cores,
        extra_executor_memory_for_ray=args.extra_executor_memory_for_ray)
    ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
    ray_ctx.init()

    runner = HorovodRayRunner(ray_ctx)
    runner.run(func=run_horovod)
