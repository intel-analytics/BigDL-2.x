import argparse

import time
import os
import cv2
from utils import streaming_image_producer
from multiprocessing import Process
from ast import literal_eval

def inqueue():
    img = cv2.imread("/opt/work/perf/7680_17920.jpg")
    img = cv2.resize(img, (224, 224))

    img = cv2.imencode(".jpg", img)[1]
    shape = str(img.shape)
    dtype = str(img.dtype)

    num = 10000
    start = time.time()
    for i in range(num):
        streaming_image_producer.image_enqueue("img2", img, img.shape, img.dtype)
        #print("push")
    end = time.time()
    fps = num / (end - start)
    print("fps:" + str(fps))

if __name__ == "__main__":
    import redis
    from utils.helpers import settings

    DB = redis.StrictRedis(host=settings.REDIS_HOST,
                           port=settings.REDIS_PORT, db=settings.REDIS_DB)
    DB.lpush("test", "xxx")
    #DB.flushall()
    while True:
        procs = []
        for i in range(1):
            proc = Process(target=inqueue)
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
        exit(0)
        time.sleep(3600)


