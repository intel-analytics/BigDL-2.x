import redis
from utils import settings, helpers
import argparse
import uuid
import time
from os import listdir
from os.path import isfile, join


DB = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def image_enqueue(fname, img, label=None):

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
    DB.xadd(settings.IMAGE_STREAMING, d)
    print("Push to redis %d micros" % int(round((time.time() - start_time) * 1000000)))

    # with open(image_path, "rb") as imageFile:
    #     # generate an ID for the classification then add the
    #     # classification ID + image to the queue
    #
    #
    #     image = helpers.base64_encode_image(imageFile.read())
    #
    #
    #     # generate an ID for the classification then add the
    #     # classification ID + image to the queue
    #     k = str(uuid.uuid4())
    #     # Streaming schema
    #     d = {"id": str(k), "path": image_path, "image": image, "label": label}
    #     DB.xadd(settings.IMAGE_STREAMING, d)
    #     print("Push to redis %d ms" % int(round((time.time() - start_time) * 1000)))


# def images_enqueue(dir_path):
#     for f in listdir(dir_path):
#         if isfile(join(dir_path, f)):
#             image_enqueue(join(dir_path, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help="Path where the images are stored")
    args = parser.parse_args()

    dir_path = args.img_path
    for f in listdir(dir_path):
        if isfile(join(dir_path, f)):
            image_enqueue(join(dir_path, f))

    # images_enqueue(args.img_path)
