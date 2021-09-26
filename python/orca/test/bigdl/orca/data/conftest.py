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
import os
from zoo.orca import OrcaContext

import pytest

sc = None
ray_ctx = None


@pytest.fixture(autouse=True, scope='package')
def orca_data_fixture():
    from zoo import init_spark_on_local
    from zoo.ray import RayContext
    OrcaContext._eager_mode = True
    sc = init_spark_on_local(cores=4, spark_log_level="INFO")
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key_id is not None and secret_access_key is not None:
        ray_ctx = RayContext(sc=sc,
                             object_store_memory="1g",
                             env={"AWS_ACCESS_KEY_ID": access_key_id,
                                  "AWS_SECRET_ACCESS_KEY": secret_access_key}
                             )
    else:
        ray_ctx = RayContext(sc=sc, object_store_memory="1g")
    ray_ctx.init()
    yield
    ray_ctx.stop()
    sc.stop()
