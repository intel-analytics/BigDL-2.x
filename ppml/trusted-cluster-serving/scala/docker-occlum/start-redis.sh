#!/bin/bash
# set -x

echo "### Launching Redis ###"

cd /opt/redis
./redis-server --tls-port $REDIS_PORT --port 0 \
    --tls-cert-file /opt/keys/server.crt \
    --tls-key-file /opt/keys/server.key \
    --tls-ca-cert-file /opt/keys/server.crt \
    --protected-mode no --maxmemory 10g | tee ./redis-sgx.log
