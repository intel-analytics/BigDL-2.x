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
