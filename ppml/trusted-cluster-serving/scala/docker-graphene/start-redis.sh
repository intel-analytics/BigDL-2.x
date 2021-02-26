#!/bin/bash
# set -x

echo "### Launching Redis ###"
KEY_PATH=/ppml/trusted-cluster-serving/redis/work/keys

cd /ppml/trusted-cluster-serving/redis
SGX=1 ./pal_loader redis-server --tls-port $REDIS_PORT --port 0 \
    --tls-cert-file $KEY_PATH/server.crt \
    --tls-key-file $KEY_PATH/server.key \
    --tls-ca-cert-file $KEY_PATH/server.crt \
    --protected-mode no --maxmemory 10g | tee ./redis-sgx.log
