 make SGX=1 GRAPHENEDIR=/home/sdp/qiyuan/redis_test/graphene         JDK_HOME=/home/sdp/opt/jdk         THIS_DIR=/home/sdp/hangrui/myzoo/analytics-zoo/ppml/trusted-big-data-ml SPARK_LOCAL_IP=$local_ip SPARK_USER=root G_SGX_SIZE=$sgx_mem_size clean
cd ~/opt/jdk/bin && \
unlink  java.sig && \
unlink  java.manifest.sgx && \
unlink  java.token

