/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive */

#ifndef _Included_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive
#define _Included_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive
 * Method:    loadOpenVinoIR
 * Signature: (Ljava/lang/String;Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive_loadOpenVinoIR
  (JNIEnv *, jobject, jstring, jstring, jint);

/*
 * Class:     com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive
 * Method:    predict
 * Signature: (J[F[I)Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive_predict
  (JNIEnv *, jobject, jlong, jfloatArray, jintArray);

/*
 * Class:     com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive
 * Method:    releaseOpenVINOIR
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_inference_OpenVinoInferenceSupportive_releaseOpenVINOIR
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
