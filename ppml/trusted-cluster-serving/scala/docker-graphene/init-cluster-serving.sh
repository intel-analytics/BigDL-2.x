#!/bin/bash
set -x

core_num=$CORE_NUM
secure_passowrd=`openssl rsautl -inkey /ppml/trusted-cluster-serving/redis/work/passowrd/key.txt -decrypt </ppml/trusted-cluster-serving/redis/work/passowrd/output.bin`
redis_host=$REDIS_HOST

sed -i "s#core_number:#core_number: ${core_num}#" config.yaml
sed -i "s#secure_enabled:#secure_enabled: true#" config.yaml
sed -i "s#secure_struct_store_password:#secure_struct_store_password: ${secure_passowrd}#" config.yaml
sed -i "s#src:#src: ${redis_host}:6379#" config.yaml
