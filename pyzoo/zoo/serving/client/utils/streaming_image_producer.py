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

import redis
from zoo.serving.client.utils import helpers
import argparse
import uuid
import time
from os import listdir
from os.path import isfile, join


# DB = redis.StrictRedis(host=settings.REDIS_HOST,
#                        port=settings.REDIS_PORT, db=settings.REDIS_DB)


def image_enqueue(fname, img, db):

    '''
    :param fname: cutted image name, e.g. 0_320.jpg
    :param img: numpy ndarray
    :param label: region label corresponded
    :return: none
    '''

    start_time = time.time()

    k = str(uuid.uuid4())
    img_encoded = helpers.base64_encode_image(img)
    d = {"id": str(k), "path": fname, "image": img_encoded}
    db.xadd("image_stream", d)
    print("Push to redis %d micros" % int(round((time.time() - start_time) * 1000000)))
