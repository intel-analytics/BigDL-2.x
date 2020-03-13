#!/usr/bin/bash

# debug flag
#set -x

HYPER=1
CPU=$(($(nproc) / HYPER))
echo "Core number ${CPU}"

usage()
{
    echo "usage:
       1. type, tf, torch, bigdl or ov, e,g., bigdl
       2. Model path, e.g., *.model, *.xml
       3. Iteration, e.g., 100
       4. Batch Size, e.g., 32
       5. blas, turn on mklblas
       as parameters in order. More concretely, you can run this command:
       bash run.sh \\
            bigdl \\
            /path/model \\
            100 \\
            32"
    exit 1
}

if [ "$#" -lt 4 ]
then
    usage
else
    TYPE="$1"
    MODEL="$2"
    ITER="$3"
    BS="$4"
fi

if [ "$#" -gt 4 ]
then
    BACKEND="$5"
fi

case $TYPE in

  "tf" | "TF")
    CLASS=com.intel.analytics.zoo.benchmark.inference.TFNetPerf
    ;;

  "torch" | "TORCH")
    CLASS=com.intel.analytics.zoo.benchmark.inference.TorchNetPerf
    export OMP_NUM_THREADS=${CPU}
    ;;

  "bigdl" | "BIGDL")
    CLASS=com.intel.analytics.zoo.benchmark.inference.BigDLPerf
    OPTIONS='-Dbigdl.engineType=mkldnn -Dbigdl.mklNumThreads='${CPU}
    ;;

  "bigdlblas" | "BIGDLBLAS")
    CLASS=com.intel.analytics.zoo.benchmark.inference.BigDLBLASPerf
    PARM="-c ${CPU}"
    ;;

  "ov" | "OV")
    CLASS=com.intel.analytics.zoo.benchmark.inference.OpenVINOPerf
    export OMP_NUM_THREADS=${CPU}
    export KMP_BLOCKTIME=20
    ;;

  *)
    CLASS=com.intel.analytics.zoo.benchmark.inference.BigDLPerf
    OPTIONS='-Dbigdl.engineType=mkldnn -Dbigdl.mklNumThreads='${CPU}
    ;;
esac

if [[ -z "${BACKEND}" ]]; then
    echo "Using mkldnn"
else
    echo "Using mklblas"
    OPTIONS=""
fi


# for maven
JAR=target/benchmark-0.1.0-SNAPSHOT-jar-with-dependencies.jar

java ${OPTIONS} -cp ${JAR} ${CLASS} -m ${MODEL} --iteration ${ITER} --batchSize ${BS} ${PARM}
