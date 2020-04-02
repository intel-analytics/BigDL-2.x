#!/bin/bash

# To validate imagenet by one command
# you have to specify image path and total image number
# e.g. imagenet_correctness.sh /path/to/imagenet_dir 50000

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "you must specify image path and total number"
  exit 1
fi

chmod a+x ./*
fuser -k 6006/tcp
rm cluster-serving.log
rm -rf TensorboardEventLogs
rm running &&
echo "old logs removed"

export ANALYTICS_ZOO_HOME=$(pwd)/../../../../..
export PYTHONPATH=$PYTHONPATH:$ANALYTICS_ZOO_HOME/pyzoo
export BPATH=$ANALYTICS_ZOO_HOME/scripts/cluster-serving

python3 $BPATH/cluster-serving-init &&
echo "initialized env" &&
$BPATH/cluster-serving-start &
sleep 10

python3 ./imagenet_enqueuer.py --img_path $1 --img_num $2

$BPATH/cluster-serving-stop

echo "Validation ended"
