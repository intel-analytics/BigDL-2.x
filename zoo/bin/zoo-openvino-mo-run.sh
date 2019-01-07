#!/bin/bash

usage()
{
    echo "usage: You have to provide
       1.the frozen inference model (.pb),
       2.the pipeline config file of the model (pipeline.config or so),
       3.the extensions config file supporting the model (.json),
       4.the output directory of generated Openvino IR
       as four parameters in order. More concretely, you can run this command:
       sh zoo-mo-run.sh /path/to/the/frozen/model \\
                        /path/to/the/pipeline/config/file \\
                        /path/to/the/extensions/config/file \\
                        /the/output/directory
       You can find available extensions config files in this folder: model_optimizer/extensions/front/tf"
    exit 1
}

if [ "$#" -ne 4 ]
then
    usage
else
    FROZEN_MODEL="$1"
    PIPELINE_CONFIG_FILE="$2"
    EXTENSIONS_CONFIG_FILE="$3"
    OPENVINO_IR_OUTPUT_DIR="$4"
fi

cd /opt/zoo-openvino/model_optimizer
python3 mo_tf.py \
--input_model ${FROZEN_MODEL} \
--tensorflow_object_detection_api_pipeline_config ${PIPELINE_CONFIG_FILE} \
--tensorflow_use_custom_operations_config ${EXTENSIONS_CONFIG_FILE} \
--output_dir ${OPENVINO_IR_OUTPUT_DIR}