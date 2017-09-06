#!/bin/bash
ssd_root=${HOME}/analytics-zoo/pipeline/ssd
data_root=${ssd_root}/data/models

cd $data_root
if [ ! -d "VGGNet" ]; then
  mkdir VGGNet
fi

# download fully convolutional reduced (atrous) VGGNet
cd VGGNet
wget http://cs.unc.edu/~wliu/projects/ParseNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
wget https://gist.githubusercontent.com/weiliu89/2ed6e13bfd5b57cf81d6/raw/758667b33d1d1ff2ac86b244a662744b7bb48e01/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt



