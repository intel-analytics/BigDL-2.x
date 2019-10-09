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

from zoo.serving.client.utils import streaming_image_producer, config_parser


def push_to_redis(id, data):
    """
    :param id: String you use to identify this record
    :param data: Data, ndarray type
    :return:
    """

    streaming_image_producer.image_enqueue(id, data, DB)


def get_from_redis(id):
    return DB.hgetall(id)


if __name__ == "__main__":
    # if you do not specify file_path, it
    # file_path = "/path/to/analytics-zoo-cluster-serving/config.yaml"
    file_path = None

    cfg = config_parser.Config(file_path)
    DB = cfg.db

    '''
    following lines demonstrate how to push data to redis
    '''
    # img_path = "/path/to/image"
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (224, 224))
    # data = cv2.imencode(".jpg", img)[1]
    # for i in range(1):
    #     push_to_redis(img_path, data)

    '''
    following lines demonstrate how to get data from redis
    '''
    # res_list = DB.keys('result:*')
    # for res in res_list:
    #
    #     res_dict = (get_from_redis(res.decode('utf-8')))
    #     res_id = res_dict[b'id'].decode('utf-8')
    #     res_value = res_dict[b'value'].decode('utf-8')
    #     print('data:', res_id,
    #           ' result:', res_value)
