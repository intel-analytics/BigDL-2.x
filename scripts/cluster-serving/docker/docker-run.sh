#!/bin/bash
docker run -itd --name cluster-serving --net=host -v $(pwd)/../model:/opt/work/model -v $(pwd)/../config.yaml:/opt/work/config.yaml intelanalytics/zoo-cluster-serving:beta
