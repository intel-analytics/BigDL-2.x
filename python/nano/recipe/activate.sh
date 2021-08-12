#!/bin/bash -e -x

echo "=====activate===="
is_tf = $(conda list orca-lite.tf | grep orca-lite.tf) || is_tf = "no"

if [ "${is_tf}" = "no" ];then
    echo "Use orca-lite-init support for pytorch"
    orca-lite-init
else
    echo "find orca-lite-init support for tensorflow"
    orca-lite-init -tf
fi
