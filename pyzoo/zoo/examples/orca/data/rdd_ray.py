import ray
from zoo import init_spark_on_yarn
from zoo.ray import RayContext

sc = init_spark_on_yarn(
    hadoop_conf="/opt/work/hadoop-2.7.2/etc/hadoop",
    conda_name="kai-hyperseg",
    num_executor=2,
    executor_cores=44,
    executor_memory="10g",
    driver_memory="2g",
    driver_cores=16,
    extra_executor_memory_for_ray="5g")
ray_ctx = RayContext(sc=sc, object_store_memory="10g")
ray_info = ray_ctx.init()
plasma_address = ray_info["object_store_address"]


@ray.remote(num_cpus=22)
class Runner(object):
    def setup_plasma(self, store_socket_name):
        import pyarrow.plasma as plasma
        self.client = plasma.connect(store_socket_name)

    def get_data(self, ids):
        self.data = self.client.get(ids)
        return self.data

    def get_node_ip(self):
        import ray.services
        return ray.services.get_node_ip_address()


def put_to_object_store(splitIndex, iterator):
    import random
    import pyarrow.plasma as plasma
    import socket
    random.seed(splitIndex)
    res = list(iterator)
    client = plasma.connect(plasma_address)
    object_id = client.put(res)
    # Not directly calling ray.services.get_node_ip_address()
    # since this would introduce ray overhead.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    node_ip = s.getsockname()[0]
    s.close()
    yield object_id, node_ip


rdd = sc.parallelize(range(1, 100000), numSlices=4)
id_ips = rdd.mapPartitionsWithIndex(put_to_object_store).collect()
id_ips.sort(key=lambda x: x[1])
print(id_ips)

runners = [Runner.remote() for i in range(4)]
print("Runners created")
runner_ips = ray.get([runner.get_node_ip.remote() for runner in runners])
runner_ips = list(zip(runners, runner_ips))
runner_ips.sort(key=lambda x: x[1])
print(runner_ips)
ray.get([runner.setup_plasma.remote(plasma_address) for runner in runners])
print("Plasma connected")
retrieved = ray.get([runner_ip[0].get_data.remote(id_ips[i][0]) for i, runner_ip in enumerate(runner_ips)])
for data in retrieved:
    print(len(data), data[0])

print("Finished")

ray_ctx.stop()
sc.stop()
