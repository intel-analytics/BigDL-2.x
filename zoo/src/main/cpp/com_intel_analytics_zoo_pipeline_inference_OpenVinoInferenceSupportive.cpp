#include <jni.h>
#include <iostream>
#include <string>

#include "common/samples/slog.hpp"
#include "inference_engine/inference_engine.hpp"
#include "OpenVINOInferenceSupportive.hpp"
#include "extension/ext_list.hpp"

#include "com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive.h"

using namespace InferenceEngine;

extern "C" {
    JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive_loadOpenVinoIR
        (JNIEnv * env, jobject thisObj, jstring modelFilePath, jstring weightFilePath, jint deviceType)
        {
            const char* model_path = env->GetStringUTFChars(modelFilePath, NULL);
            std::string c_model_path(model_path);
            const char* weight_path = env->GetStringUTFChars(weightFilePath, NULL);
            std::string c_weight_path(weight_path);
            int c_device_type = deviceType;

            ExecutableNetwork * executableNetwork = OpenVINOInferenceSupportive::loadOpenVINOIR(c_model_path, c_weight_path, c_device_type);

            long res = (long) executableNetwork;
            env->ReleaseStringUTFChars(modelFilePath, model_path);
            env->ReleaseStringUTFChars(weightFilePath, weight_path);
            return res;
        }

    JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive_predict
        (JNIEnv * env, jobject thisObj, jlong executableNetworkReference, jfloatArray data, jintArray shape)
        {
            ExecutableNetwork* executableNetwork =  (ExecutableNetwork*) ((long)executableNetworkReference);
            int shape_length = env->GetArrayLength(shape);
            jfloat* c_tensor_data = (jfloat*) env->GetPrimitiveArrayCritical(data, JNI_FALSE);
            jint* c_tensor_shape = (jint*) env->GetPrimitiveArrayCritical(shape, JNI_FALSE);
            std::vector<size_t> tensor_shape;
            tensor_shape.assign(c_tensor_shape, c_tensor_shape + shape_length);
            CTensor<float> input_tensor((float*) c_tensor_data, tensor_shape);

            CTensor<float> predict_output = OpenVINOInferenceSupportive::predict(*executableNetwork, input_tensor);

            env->ReleasePrimitiveArrayCritical(data, c_tensor_data, 0);
            env->ReleasePrimitiveArrayCritical(shape, c_tensor_shape, 0);
            jclass jtensor_class = env->FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
            jmethodID jtensor_init = env->GetMethodID(jtensor_class, "<init>", "([F[I)V");
            jfloatArray out_floats = env->NewFloatArray(predict_output.data_size);
            env->SetFloatArrayRegion(out_floats, 0, predict_output.data_size,  predict_output.data);
            int *predict_output_shape = new int(predict_output.shape.size());
            for(int i = 0; i < predict_output.shape.size(); i++)
            {
                predict_output_shape[i] = predict_output.shape[i];
            }
            jintArray out_ints = env->NewIntArray(predict_output.shape.size());
            env->SetIntArrayRegion(out_ints, 0, predict_output.shape.size(), predict_output_shape);
            jobject result = env->NewObject(jtensor_class, jtensor_init, out_floats, out_ints);
            return result;
        }

    JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive_releaseOpenVINOIR
        (JNIEnv * env, jobject thisObj, jlong executableNetworkReference)
        {
            ExecutableNetwork* executableNetwork =  (ExecutableNetwork*) ((long)executableNetworkReference);
            OpenVINOInferenceSupportive::destoryExecutableNetworkPtr(executableNetwork);
            return;
        }
}