import tensorflow as tf

# _RESNET_LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
#     (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
# ]
# _RESNET_LR_BOUNDARIES = list(p[1] for p in _RESNET_LR_SCHEDULE[1:])
# _RESNET_LR_MULTIPLIERS = list(p[0] for p in _RESNET_LR_SCHEDULE)
# _RESNET_LR_WARMUP_EPOCHS = _RESNET_LR_SCHEDULE[0][1]
#
# BASE_LEARNING_RATE = 0.1


def model_creator(config):
    wd = config["wd"]
    import tensorflow as tf
    import tensorflow.keras as keras
    # from tensorflow.keras.mixed_precision import experimental as mixed_precision
    # policy = mixed_precision.Policy('mixed_bfloat16')
    # policy = mixed_precision.Policy('float32')
    # mixed_precision.set_policy(policy)

    model = tf.keras.applications.resnet50.ResNet50(weights=None)
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = tf.keras.models.Model.from_config(model_config)
    return model


def compile_args_creator(config):
    momentum = config["momentum"]
    num_worker = config["num_worker"]
    lr = config["lr"]
    import tensorflow.keras as keras
    opt = keras.optimizers.SGD(learning_rate=lr * num_worker,
                               momentum=momentum)
    param = dict(loss=keras.losses.categorical_crossentropy, optimizer=opt,
                 metrics=['accuracy', 'top_k_categorical_accuracy'])
    return param

def data_creator2(config):
    from .imagenet_preprocessing import input_fn
    train_dataset = input_fn(is_training=True,
             data_dir=config["data_dir"],
             batch_size=config["batch_size"])

    val_dataset = input_fn(is_training=False,
                           data_dir=config["data_dir"],
                           batch_size=config["batch_size"])

    return train_dataset, val_dataset


def data_creator(config):
    train_dir = config["train_dir"]
    batch_size = config["batch_size"]
    val_dir = config["val_dir"]
    val_batch_size = config["val_batch_size"]
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import tensorflow.keras as keras
    train_gen = image.ImageDataGenerator(
        width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
        preprocessing_function=keras.applications.resnet50.preprocess_input)
    train_iter = train_gen.flow_from_directory(train_dir,
                                               batch_size=batch_size,
                                               target_size=(224, 224))

    # Validation data iterator.
    test_gen = image.ImageDataGenerator(
        zoom_range=(0.875, 0.875), preprocessing_function=keras.applications.resnet50.preprocess_input)
    test_iter = test_gen.flow_from_directory(val_dir,
                                             batch_size=val_batch_size,
                                             target_size=(224, 224))

    train_dataset = tf.data.Dataset.from_generator(lambda: train_iter, (tf.float32, tf.int32),
                                                   output_shapes=((batch_size, 224, 224, 3), (batch_size, 1000)))
    val_dataset = tf.data.Dataset.from_generator(lambda: test_iter, (tf.float32, tf.int32),
                                                 output_shapes=((val_batch_size, 224, 224, 3), (val_batch_size, 1000)))

    return train_dataset, val_dataset


import argparse
from zoo.ray import RayContext
from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.orca.learn.tf.tf_ray_estimator import TFRayEstimator

parser = argparse.ArgumentParser()
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                         " configuration folder. Otherwise, turn on local mode.")
parser.add_argument("--slave_num", type=int, default=2,
                    help="The number of slave nodes")
parser.add_argument("--conda_name", type=str, default=None,
                    help="The name of conda environment.")
parser.add_argument("--penv_archive", type=str, default=None)
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
parser.add_argument("--archive", type=str)
parser.add_argument("--archive_name", type=str)

if __name__ == "__main__":

    args = parser.parse_args()
    if args.hadoop_conf:
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name=args.conda_name,
            penv_archive=args.penv_archive,
            num_executors=args.slave_num,
            executor_cores=args.executor_cores,
            executor_memory=args.executor_memory,
            driver_memory=args.driver_memory,
            driver_cores=args.driver_cores,
            additional_archive=args.archive + "#" + args.archive_name,
            extra_executor_memory_for_ray=args.extra_executor_memory_for_ray)
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()
    else:
        sc = init_spark_on_local(cores=8)
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()

        # ray_ctx.ray_node_cpu_cores = 4
        # ray_ctx.num_ray_nodes = 2

    num_workers = ray_ctx.num_ray_nodes

    def schedule(epoch):
        if epoch <= 5:
            return 0.1 * num_workers * epoch / 5

        if 5 < epoch <= 30:
            return 0.1 * num_workers

        if 30 < epoch <= 60:
            return 0.1 * num_workers * 0.1

        if 60 < epoch <= 80:
            return 0.1 * num_workers * 0.01

        return 0.1 * num_workers * 0.001

    lr_schdule = tf.keras.callbacks.LearningRateScheduler(schedule)

    config = {
        "lr": 0.1,
        "momentum": 0.9,
        "wd": 0.00005,
        "batch_size": 256,
        "val_batch_size": 256,
        "warmup_epoch": 5,
        "num_worker": ray_ctx.num_ray_nodes,
        "train_dir": "/home/yang/sources/datasets/imagenet-raw-image/imagenet-2012-small/train",
        "val_dir": "/home/yang/sources/datasets/imagenet-raw-image/imagenet-2012-small/train",
        "data_dir": "hdfs:///yang/imagenet_tf_record",
        "fit_config": {
            "steps_per_epoch": 1280000 // (256 * num_workers),
            "callbacks": [lr_schdule]
        },
        "evaluate_config": {
            "steps": (50000 // (256 * num_workers)),
        }
    }
    trainer = TFRayEstimator(
        model_creator=model_creator,
        compile_args_creator=compile_args_creator,
        verbose=True,
        config=config, backend="horovod")

    print(trainer.fit(data_creator2))
