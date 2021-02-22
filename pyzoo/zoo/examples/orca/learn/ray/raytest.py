import argparse
from zoo.orca import init_orca_context

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local or yarn.')
args = parser.parse_args()
cluster_mode = args.cluster_mode
if cluster_mode == "local":
    sc = init_orca_context(cluster_mode="local", cores=4, init_ray_on_spark=True)
elif cluster_mode == "yarn":
    sc = init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=4,
                           memory="10g",
                           init_ray_on_spark=True
                           )
else:
    print("init_orca_context failed. cluster_mode should be either 'local' or 'yarn' but got "
          + cluster_mode)

import ray


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1
        return self.n


counters = [Counter.remote() for i in range(5)]
print(ray.get([c.increment.remote() for c in counters]))
from zoo.orca import OrcaContext

ray_ctx = OrcaContext.get_ray_context()
# The dictionary information of the ray cluster,
# including node_ip_address, object_store_address, webui_url, etc.
address_info = ray_ctx.address_info
# The redis address of the ray cluster.
redis_address = ray_ctx.redis_address
from zoo.orca import stop_orca_context

stop_orca_context()
