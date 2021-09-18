from os import system

# build zoo package -> install zoo package -> install and start Redis and Flink -> start cluster serving
system("../../zoo/make-dist.sh && \
        echo install zoo package && \
        cd ../../pyzoo && \
        pwd && \
        python setup.py sdist && \
        pip install dist/*.tar.gz && \
        cd .. && \
        mkdir tmp && \
        echo REDIS && \
        cd tmp && \
        export REDIS_VERSION=5.0.5 && \
        wget http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
        tar xzf redis-${REDIS_VERSION}.tar.gz && \
        rm redis-${REDIS_VERSION}.tar.gz && \
        cd redis-${REDIS_VERSION} && \
        make && \
        export REDIS_HOME=$(pwd) && \
        echo install flink && \
        cd .. && \
        export FLINK_VERSION=1.11.2 && \
        wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
        tar xzf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
        rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
        cd flink-${FLINK_VERSION} && \
        export FLINK_HOME=$(pwd) && \
        $FLINK_HOME/bin/start-cluster.sh && \
        $FLINK_HOME/bin/flink list && \
        cd ../.. && \
        echo start cluster serving && \
        cd dist/bin/cluster-serving && \
        pwd && \
        bash cluster-serving-init && \
        bash cluster-serving-start && \
        echo CHECK_FLINK && \
        $FLINK_HOME/bin/flink list && \
        rm -r ../../../tmp ")


# predict
print("predict")
#system("conda  list")
from zoo.serving.client import *

#system("conda env list")
import numpy
import cv2
input_api = InputQueue()
path="/home/qihong/Documents/test.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (224, 224))
data = cv2.imencode(".jpg", img)[1]
img_encoded = base64.b64encode(data).decode("utf-8")
input_api.enqueue("my-image2", t={"b64": img_encoded})
import time
time.sleep(3)
print("output")
output_api = OutputQueue()
result_ndarray = output_api.query("my-image2")
print(result_ndarray)