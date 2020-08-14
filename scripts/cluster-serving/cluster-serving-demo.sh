#!/bin/bash

source ./cluster-serving-setup.sh
./cluster-serving-start -c ./config.yaml & sleep 20
python3 ./quick_start.py --image_path ../test_image

sleep 20
./cluster-serving-stop

echo "Demo end, please check at flink Dashboard"
