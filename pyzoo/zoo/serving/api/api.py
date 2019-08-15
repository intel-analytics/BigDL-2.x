import redis
import cv2
import yaml
from utils import streaming_image_producer, config_parser


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

    file_path = "/home/litchy/pro/analytics-zoo-serving/serving/config.yaml"

    cfg = config_parser.Config(file_path)
    DB = cfg.db

    img = cv2.imread("/home/litchy/health/test_image/7680_17920.jpg")
    img = cv2.resize(img, (224, 224))
    data = cv2.imencode(".jpg", img)[1]
    for i in range(10000):
        push_to_redis("img1", data)
    pass
