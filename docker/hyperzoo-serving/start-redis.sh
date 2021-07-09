# start redis
${REDIS_HOME}/src/redis-server --daemonize yes --port ${REDIS_PORT} --dir ${REDIS_STORAGE} --protected-mode no --maxmemory 10g
