import tensorflow as tf
from bigdl.nano.common.cpu_schedule import schedule_workers, get_cpu_info

proc_list = schedule_workers(1)
_, get_socket = get_cpu_info()

num_sockets = len(set(get_socket.values()))
num_threads = len(proc_list[0]) // num_sockets

tf.config.threading.set_inter_op_parallelism_threads(num_sockets)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(enabled=True)
