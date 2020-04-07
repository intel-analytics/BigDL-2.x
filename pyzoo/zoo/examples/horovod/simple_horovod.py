import argparse
import ray
from zoo import init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext
from horovod.run.gloo_run import RendezvousServer, _allocate
from horovod.run import _get_driver_ip
from horovod.run.common.util.config_parser import set_env_from_args

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


    @ray.remote(num_cpus=args.executor_cores)
    class HorovodWorker():
        def hostname(self):
            import socket
            return socket.gethostname()

        # todo maybe create another process and
        # set the env and run horovod in that process
        # without polluting the process's environment
        def set_env(self, envs):
            import os
            os.environ.update(envs)

        def run(self, func):
            return func()

    actors = [HorovodWorker.remote() for i in range(0, args.slave_num)]
    hosts = ray.get([actor.hostname.remote() for actor in actors])

    host_to_size = {}
    host_and_rank_to_actor_idx = {}
    for i, host in enumerate(hosts):
        if host not in host_to_size:
            host_to_size[host] = 0
        else:
            host_to_size[host] = host_to_size[host] + 1
        host_and_rank_to_actor_idx[(host, host_to_size[host])] = i

    hosts_spec = ["{}:{}".format(key, host_to_size[key]) for key in host_to_size]
    host_alloc_plan = _allocate(",".join(hosts_spec), args.slave_num)
    global_rendezv = RendezvousServer(True)
    global_rendezv_port = global_rendezv.start_server(host_alloc_plan)

    # todo args.iface should be inferenced automatically
    # instead of letting user pass in
    driver_ip = _get_driver_ip([args.iface])
    envs = {
        "HOROVOD_GLOO_RENDEZVOUS_ADDR": driver_ip,
        "HOROVOD_GLOO_RENDEZVOUS_PORT": global_rendezv_port,
        "HOROVOD_CONTROLLER": "gloo",
        "HOROVOD_CPU_OPERATIONS": "gloo",
        "HOROVOD_GLOO_IFACE": args.iface,
        "PYTHONUNBUFFERED": '1',
    }

    set_env_from_args(envs, args)

    ids = []
    for alloc_info in host_alloc_plan:
        local_envs = envs.copy()
        local_envs["HOROVOD_RANK"] = alloc_info.rank
        local_envs["HOROVOD_SIZE"] = alloc_info.size
        local_envs["HOROVOD_LOCAL_RANK"] = alloc_info.local_rank
        local_envs["HOROVOD_LOCAL_SIZE"] = alloc_info.local_size
        local_envs["HOROVOD_CROSS_RANK"] = alloc_info.cross_rank
        local_envs["HOROVOD_CROSS_SIZE"] = alloc_info.cross_size

        key = (alloc_info.host_name, alloc_info.local_rank)
        actor = actors[host_and_rank_to_actor_idx[key]]
        ids.append(actor.set_env.remote(local_envs))

    ray.wait(ids)

    results = ray.get([actor.run.remote(run_horovod) for actor in actors])










