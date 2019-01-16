#!/bin/bash

set -x

echo "### start to prepare header files for zoo-openvino"
export openvino_version=2018_R3
wget https://github.com/opencv/dldt/archive/$openvino_version.tar.gz
tar -xzvf $openvino_version.tar.gz
rm $openvino_version.tar.gz
mkdir include
mkdir -p ./include/common/samples
mkdir -p ./include/inference_engine
cp -r ./dldt-$openvino_version/inference-engine/src/extension ./include
cp ./dldt-$openvino_version/inference-engine/samples/common/samples/slog.hpp ./include/common/samples
cp -rf ./dldt-$openvino_version/inference-engine/include/*  ./include/inference_engine

echo "### start to prepare model-optmizer for openvino"
tar -xzvf ../../src/main/resources/inference.tar.gz -C ../classes
cp -rf ./dldt-$openvino_version/model-optimizer ../classes/model_optimizer

echo "### start to prepare pipeline-configs"
export tf_model_branch=master
mkdir -p pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/embedded_ssd_mobilenet_v1_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/facessd_mobilenet_v2_quantized_320x320_open_image_v4.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_oid.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_inception_v2_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_nas_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet101_atrous_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet101_ava_v2.1.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet101_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet101_fgvc.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet101_kitti.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet101_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet101_voc07.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet152_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet152_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet50_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet50_fgvc.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/faster_rcnn_resnet50_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/mask_rcnn_inception_resnet_v2_atrous_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/mask_rcnn_inception_v2_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/mask_rcnn_resnet101_atrous_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/mask_rcnn_resnet101_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/mask_rcnn_resnet50_atrous_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/rfcn_resnet101_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/rfcn_resnet101_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_inception_v2_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_inception_v2_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_inception_v3_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_300x300_coco14_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_focal_loss_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v1_quantized_300x300_coco14_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_mobilenet_v2_quantized_300x300_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssdlite_mobilenet_v1_coco.config -P pipeline-configs/object_detection
wget https://raw.githubusercontent.com/tensorflow/models/$tf_model_branch/research/object_detection/samples/configs/ssdlite_mobilenet_v2_coco.config -P pipeline-configs/object_detection
mv pipeline-configs ../classes/
