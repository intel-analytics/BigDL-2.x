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

def get_ray_node_resources_info():
    import ray
    driver_ip = ray._private.services.get_node_ip_address()
    resources = ray.cluster_resources()
    nodes = []
    for key, value in resources.items():
        if key.startswith("node:"):
            # if running in cluster, filter out driver ip
            if key != f"node:{driver_ip}":
                nodes.append(key)
    # for the case of local mode and single node spark standalone
    if not nodes:
        nodes.append(f"node:{driver_ip}")
