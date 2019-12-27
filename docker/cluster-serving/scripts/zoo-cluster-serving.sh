#!/bin/bash

# --------------config


/opt/work/redis-5.0.5/src/redis-server > /opt/work/redis.log &
echo "redis server started, please check log in /opt/work/redis.log" && sleep 1

# sleep for 1 sec to ensure server is ready and client could connect
/opt/work/redis-5.0.5/src/redis-cli config set stop-writes-on-bgsave-error no
/opt/work/redis-5.0.5/src/redis-cli config set save ""
if [ -z "${REDIS_MAXMEM}" ]; then
    echo "Redis maxmemory is not set, using default value 8G"
    REDIS_MAXMEM=8G
fi
/opt/work/redis-5.0.5/src/redis-cli config set maxmemory ${REDIS_MAXMEM}
echo "redis config maxmemory set to ${REDIS_MAXMEM}"
# bind can not be set after redis starts
# /opt/work/redis-5.0.5/src/redis-cli config set bind "0.0.0.0"
/opt/work/redis-5.0.5/src/redis-cli config set protected-mode no
/opt/work/redis-5.0.5/src/redis-cli config set maxclients 10000

tensorboard --logdir . --bind_all &

./start-cluster-serving.sh &

/bin/bash
