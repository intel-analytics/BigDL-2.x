from zoo.serving.client.api import RedisQueue
import numpy as np


if __name__ == "__main__":
    redis_queue = RedisQueue()
    # no img
    redis_queue.enqueue_image("img1")
    # wrong format
    redis_queue.enqueue_image("img1", np.zeros([3, 224, 225]))
