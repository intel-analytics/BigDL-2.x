#!/bin/bash

usage()
{
    echo "usage: Required python version should be greater or equal than 3.4"
    exit 1
}

#check user's python version
V1=3
V2=4
U_V1=`python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1}'`
U_V2=`python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}'`
U_V3=`python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $3}'`
if [ ! $V1=$U_V1 ]
then
    usage
elif [ $V2 -gt $U_V2 ]
then
    usage
else
    echo "Requirement already satisfied: python>=3.4 ($U_V1.$U_V2.$U_V3) "
fi

#install packages
if [ `whoami` = "root" ]
then
    pip3 install --upgrade pip
    pip3 install numpy
    pip3 install networkx
    pip3 install tensorboard==1.12.0
    pip3 install tensorflow==1.12.0
else
    sudo pip3 install --upgrade pip
    sudo pip3 install numpy
    sudo pip3 install networkx
    sudo pip3 install tensorboard==1.12.0
    sudo pip3 install tensorflow==1.12.0
fi

#download model optimizer code
if [ ! -d "/opt" ]
then
    mkdir /opt
fi
if [ ! -d "/opt/zoo-openvino" ]
then
    cd /opt
    mkdir zoo-openvino
fi
cd /opt/zoo-openvino
if [ ! -d "./model_optimizer" ]
then
    if [ `whoami` = "root" ]
    then
        apt-get install git
        git clone https://github.com/opencv/dldt.git
        cd dldt
        git checkout tags/2018_R3 -b 2018_R3
        mv ./model-optimizer ../model_optimizer
        cd ..
        rm -r dldt
    else
        sudo apt-get install git
        git clone https://github.com/opencv/dldt.git
        cd dldt
        git checkout tags/2018_R3 -b 2018_R3
        sudo mv ./model-optimizer ../model_optimizer
        cd ..
        sudo rm -r dldt
    fi
    if [ -d "./model_optimizer" ]
    then
        echo "Requirement: model_optimizer saved in /opt/zoo-openvino/"
    else
        echo "Failed to save model_optimizer"
    fi
else
    echo "Requirement: model_optimizer already exists in /opt/zoo-openvino/"
fi
