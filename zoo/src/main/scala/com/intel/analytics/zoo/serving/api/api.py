import argparse
from pyspark import SparkContext, rdd
from pyspark.streaming import StreamingContext

import time
import os
import cv2
from utils import streaming_image_producer
from multiprocessing import Process


def push_to_redis(id, data):
    """
    :param id: String you use to identify this record
    :param data: Data, ndarray type
    :return:
    """
    streaming_image_producer.image_enqueue(id, data)


if __name__ == "__main__":
    pass
