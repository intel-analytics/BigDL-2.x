#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'
conf_dir=/opt/conf

id=$([ -f "$pid" ] && echo $(wc -l < "$pid") || echo "0")
FLINK_LOG_PREFIX="/host/flink--$postfix-${id}"
log="${FLINK_LOG_PREFIX}.log"

run_taskmanager() {
    pushd /opt/flink
    #if conf_dir exists, use the new configurations.
    if [ -d $conf_dir  ];then
        cp -r $conf_dir/* image/bin/conf/
	occlum build
	#rm -rf $conf_dir
    fi

    echo -e "${BLUE}occlum run JVM taskmanager${NC}"
    echo -e "${BLUE}logfile=$log${NC}"
	# start task manager in occlum
    occlum run /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
	-XX:+UseG1GC -Xmx25g -Xms6g -XX:MaxDirectMemorySize=5g -XX:MaxMetaspaceSize=4g \
	-Dos.name=Linux \
	-Dcom.intel.analytics.zoo.core.openvino.OpenvinoNativeLoader.DEBUG \
	-XX:ActiveProcessorCount=20 \
	-Dlog.file=$log \
	-Dlog4j.configuration=file:/bin/conf/log4j.properties \
	-Dlogback.configurationFile=file:/bin/conf/logback.xml \
	-classpath /bin/lib/flink-table-blink_2.11-1.10.1.jar:/bin/lib/flink-table_2.11-1.10.1.jar:/bin/lib/log4j-1.2.17.jar:/bin/lib/slf4j-log4j12-1.7.15.jar:/bin/lib/flink-dist_2.11-1.10.1.jar org.apache.flink.runtime.taskexecutor.TaskManagerRunner \
	-Dorg.apache.flink.shaded.netty4.io.netty.tryReflectionSetAccessible=true \
	-Dorg.apache.flink.shaded.netty4.io.netty.eventLoopThreads=20 \
	-Dcom.intel.analytics.zoo.shaded.io.netty.tryReflectionSetAccessible=true \
	--configDir /bin/conf \
	-D taskmanager.memory.framework.off-heap.size=256mb \
	-D taskmanager.memory.network.max=1024mb \
	-D taskmanager.memory.network.min=1024mb \
	-D taskmanager.memory.framework.heap.size=256mb \
	-D taskmanager.memory.managed.size=10g \
	-D taskmanager.cpu.cores=1.0 \
	-D taskmanager.memory.task.heap.size=12gb \
	-D taskmanager.memory.task.off-heap.size=1024mb &

    popd
}

run_taskmanager
