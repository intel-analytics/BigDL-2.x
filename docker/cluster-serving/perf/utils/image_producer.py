import redis
from utils import settings, helpers
import argparse
import json
import uuid
import time
from os import listdir
from os.path import isfile, join


DB = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def image_enqueue(image_path):
    start_time = time.time()
    with open(image_path, "rb") as imageFile:
        # generate an ID for the classification then add the
        # classification ID + image to the queue
        k = str(uuid.uuid4())
        image = helpers.base64_encode_image(imageFile.read())
        d = {"id": k, "path": image_path, "image": image}
        DB.rpush(settings.IMAGE_QUEUE, json.dumps(d))
        print("Push to redis %d ms" % int(round((time.time() - start_time) * 1000)))


def images_enqueue(dir_path):
    for f in listdir(dir_path):
        if isfile(join(dir_path, f)):
            image_enqueue(join(dir_path, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help="Path where the images are stored")
    args = parser.parse_args()
    images_enqueue(args.img_path)
