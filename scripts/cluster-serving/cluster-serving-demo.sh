#!/bin/bash

chmod a+x ./*

export PYTHONPATH=$PYTHONPATH:./analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.9.0-SNAPSHOT-python-api.zip

mv analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.9.0-SNAPSHOT-serving.jar zoo.jar

./cluster-serving-start -c ./config.yaml & sleep 20

python3 ./quick_start.py --image_path ../test_image

sleep 20
./cluster-serving-stop

echo "Demo end, please check at flink Dashboard"