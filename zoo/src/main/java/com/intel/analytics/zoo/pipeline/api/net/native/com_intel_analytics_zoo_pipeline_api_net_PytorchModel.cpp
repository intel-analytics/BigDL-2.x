/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <string>
#include <iostream>
#include <torch/script.h>
#include <memory>

#include "com_intel_analytics_zoo_pipeline_api_net_PytorchModel.h"
using namespace std;

extern "C" {

std::mutex mtx;
std::unordered_map<int, std::shared_ptr<torch::jit::script::Module>> modelHandles;
std::unordered_map<int, std::shared_ptr<torch::jit::script::Module>> lossHandles;
std::unordered_map<int, c10::IValue> modelInputs; // always store tuple
std::unordered_map<int, c10::IValue> modelOutputs; // tensor or tuple, depending on the model
std::unordered_map<int, c10::IValue> lossGrads; // always store tuple
long modelID;
long lossID;


auto getWeights(std::shared_ptr<torch::jit::script::Module> m, std::vector<float> &xv) -> int {
    auto children = m->get_modules();

    if (children.size() == 0) {
        auto slots = m -> get_parameters();
        for (size_t i = 0; i < slots.size(); ++i) {
            auto& x = slots[i];
            size_t x_size = x.value().toTensor().numel();
            auto p = static_cast<float*>(x.value().toTensor().storage().data());
            for(size_t i = 0; i < x_size; i++)
            {
                xv.push_back(p[i]);
            }
        }
    } else {
        for (const auto& child : children) {
            getWeights(child, xv);
        }
    }

    return 1;
}

auto getAndZeroGradients(std::shared_ptr<torch::jit::script::Module> m, std::vector<float> &xv) -> int {
    auto children = m->get_modules();

    if (children.size() == 0) {
        auto slots = m -> get_parameters();

        for (size_t i = 0; i < slots.size(); ++i) {
            auto& x = slots[i];
            auto& grad = x.value().toTensor().grad();
            if (grad.defined()) {
                size_t x_size = grad.numel();
                auto p = static_cast<float*>(grad.storage().data());
                for(size_t i = 0; i < x_size; i++)
                {
                    xv.push_back(p[i]);
                }
                grad = grad.detach();
                grad.zero_();
            }
        }
    } else {
        for (const auto& child : children) {
            getAndZeroGradients(child, xv);
        }
    }

    return 1;
}

auto updateWeights(std::shared_ptr<torch::jit::script::Module> m, float* xv, int& index) -> int {
    auto children = m->get_modules();

    if (children.size() == 0) {
        auto slots = m -> get_parameters();
        for (size_t i = 0; i < slots.size(); ++i) {
            auto slot_tensor = slots[i].value().toTensor();
            auto num_slot_parameters = slot_tensor.nbytes() / slot_tensor.element_size();
            auto slot_shape = slot_tensor.sizes();

            auto new_tensor = torch::from_blob(xv + index, slot_shape, at::kFloat);
            slot_tensor.set_requires_grad(false);
            slot_tensor.mul_(0.0).add_(new_tensor);
            slot_tensor.set_requires_grad(true);
            index += num_slot_parameters;
        }
    } else {
        for (const auto& child : children) {
            updateWeights(child, xv, index);
        }
    }

    return 1;
}

// convert Torch Tensor to JTensor, copy data
jobject torch2JTensor(JNIEnv *jenv, at::Tensor &tensor) {
    // Wrap to Zoo JTensor
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jmethodID jtensor_constructor = jenv -> GetMethodID(jtensor_class, "<init>", "([F[I)V");

    auto sizes = tensor.sizes();
    int result_storage_len = 1;
    float *pytorch_result_storage = tensor.data<float>();
    int result_shape_len = sizes.size();
    int pytorch_result_shape[result_shape_len];
    for (int j = 0; j < result_shape_len; j++) {
        pytorch_result_shape[j] = sizes[j];
        result_storage_len *= sizes[j];
    }

    jfloatArray result_storage = jenv -> NewFloatArray(result_storage_len);
    jenv -> SetFloatArrayRegion(result_storage, 0, result_storage_len, pytorch_result_storage);

    jintArray result_shape = jenv -> NewIntArray(result_shape_len);
    jenv -> SetIntArrayRegion(result_shape, 0, result_shape_len, pytorch_result_shape);

    jobject jTensor = jenv -> NewObject(jtensor_class, jtensor_constructor, result_storage, result_shape);

    return jTensor;
}


JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_loadModelNative
  (JNIEnv *jenv, jclass jobj, jstring jmodel_path) {
    const char* p_model_path = jenv->GetStringUTFChars(jmodel_path, NULL);

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> model_ptr = torch::jit::load(p_model_path);
    assert(model_ptr != nullptr);

    mtx.lock();
    modelID++;
    long id = modelID;
    modelHandles[id] = model_ptr;
    mtx.unlock();

    jenv->ReleaseStringUTFChars(jmodel_path, p_model_path);
    return id;
  }

JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_saveModelNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jstring jmodel_path) {
    const char* p_model_path = jenv->GetStringUTFChars(jmodel_path, NULL);
    std::shared_ptr<torch::jit::script::Module> model_ptr = modelHandles[nativeRef];
    model_ptr -> save(p_model_path);
    jenv->ReleaseStringUTFChars(jmodel_path, p_model_path);
    return nativeRef;
  }

JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_loadLossNative
  (JNIEnv *jenv, jclass jobj, jstring jloss_path) {
    const char* p_loss_path = jenv->GetStringUTFChars(jloss_path, NULL);

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> loss_ptr = torch::jit::load(p_loss_path);
    assert(loss_ptr != nullptr);

    mtx.lock();
    lossID++;
    long id = lossID;
    lossHandles[id] = loss_ptr;
    mtx.unlock();

    jenv->ReleaseStringUTFChars(jloss_path, p_loss_path);
    return id;
  }

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    forward
 * Signature: ([F[I)Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_modelForwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jboolean isTraining, jobjectArray input_jstorage, jintArray input_joffset, jobjectArray input_jshape) {

    // keep track of the primitive array to release later.
    std::vector<jfloatArray> j_data_vector;
    std::vector<jfloat*> c_data_vector;
    std::vector<jintArray> j_shape_vector;
    std::vector<jint*> c_shape_vector;

    // create Input tuple
    int input_size = jenv -> GetArrayLength(input_jstorage);
    jint* c_input_offsets = (jint*) jenv -> GetIntArrayElements(input_joffset, JNI_FALSE);
    std::vector<c10::IValue> input_vector;

    for (int i = 0; i < input_size; i++) {
        jfloatArray tensor_storage = (jfloatArray)jenv->GetObjectArrayElement(input_jstorage, i);
        jfloat* input_c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(tensor_storage, JNI_FALSE);
        jintArray tensor_shape = (jintArray)jenv -> GetObjectArrayElement(input_jshape, i);
        jint* input_c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(tensor_shape, JNI_FALSE);
        int c_dim_count = jenv -> GetArrayLength(tensor_shape);

        std::vector<int64_t> input_torch_shape;
        for (int i = 0; i < c_dim_count; i++) {
            input_torch_shape.push_back(*(input_c_shape + i));
        }

        auto input_tensor = torch::from_blob(input_c_storage + c_input_offsets[i], input_torch_shape, at::kFloat);
        if (isTraining) {
            input_tensor.set_requires_grad(true);
        }

        input_vector.push_back(input_tensor);

        j_data_vector.push_back(tensor_storage);
        c_data_vector.push_back(input_c_storage);
        j_shape_vector.push_back(tensor_shape);
        c_shape_vector.push_back(input_c_shape);
    }

    // Execute the model
    std::shared_ptr<torch::jit::script::Module> model_ptr = modelHandles[nativeRef];
    assert(model_ptr != nullptr);
    auto output = model_ptr->forward(input_vector);

    if (isTraining) {
        mtx.lock();
        auto input_tuple = torch::jit::Tuple::create(input_vector);
        modelInputs[nativeRef] = input_tuple;
        modelOutputs[nativeRef] = output;
        mtx.unlock();
    }

    // TODO check if the release will affect cached modelInputs[nativeRef]
    // Release critical part
    jenv -> ReleaseIntArrayElements(input_joffset, c_input_offsets, JNI_ABORT);
    for (int i = 0; i < input_size; i++) {
        jenv -> ReleasePrimitiveArrayCritical(j_data_vector[i], c_data_vector[i], 0);
        jenv -> ReleasePrimitiveArrayCritical(j_shape_vector[i], c_shape_vector[i], 0);
    }

    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");

    // TODO release output after conversion?
    if(output.isTuple()) {
        int resultSize = output.toTuple() -> elements().size();
        jobjectArray jTensorArray = jenv -> NewObjectArray(resultSize, jtensor_class, NULL);
        auto outputTuple = output.toTuple();
        for (size_t i = 0; i < resultSize; i++) {
            auto t = outputTuple -> elements()[i].toTensor();
            jobject jt = torch2JTensor(jenv, t);
            jenv -> SetObjectArrayElement(jTensorArray, i, jt);
        }
        return jTensorArray;
    } else {
        int resultSize = 1;
        jobjectArray jTensorArray = jenv -> NewObjectArray(resultSize, jtensor_class, NULL);
        auto t = output.toTensor();
        jobject jt = torch2JTensor(jenv, t);
        jenv -> SetObjectArrayElement(jTensorArray, 0, jt);
        return jTensorArray;
    }
  }


JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_releaseModelNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    mtx.lock();
    modelHandles.erase(nativeRef);
    modelInputs.erase(nativeRef);
    modelOutputs.erase(nativeRef);
    mtx.unlock();
  }

JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_modelBackwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jobjectArray input_jstorage, jintArray input_joffset, jobjectArray input_jshape) {

    std::vector<jfloatArray> j_data_vector;
    std::vector<jfloat*> c_data_vector;
    std::vector<jintArray> j_shape_vector;
    std::vector<jint*> c_shape_vector;

    // create gradOutput tuple
    int input_size = jenv -> GetArrayLength(input_jstorage);
    jint* c_input_offsets = (jint*) jenv -> GetIntArrayElements(input_joffset, JNI_FALSE);
    std::vector<c10::IValue> input_tuple;

    for (int i = 0; i < input_size; i++) {
        jfloatArray tensor_storage = (jfloatArray)jenv->GetObjectArrayElement(input_jstorage, i);
        jfloat* input_c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(tensor_storage, JNI_FALSE);

        jintArray tensor_shape = (jintArray)jenv->GetObjectArrayElement(input_jshape, i);
        jint* input_c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(tensor_shape, JNI_FALSE);
        int c_dim_count = jenv -> GetArrayLength(tensor_shape);

        std::vector<int64_t> input_torch_shape;
        for (int i = 0; i < c_dim_count; i++) {
            input_torch_shape.push_back(*(input_c_shape + i));
        }

        auto input_tensor = torch::from_blob(input_c_storage + c_input_offsets[i], input_torch_shape, at::kFloat);

        input_tuple.push_back(input_tensor);

        j_data_vector.push_back(tensor_storage);
        c_data_vector.push_back(input_c_storage);
        j_shape_vector.push_back(tensor_shape);
        c_shape_vector.push_back(input_c_shape);
    }

    auto y = modelOutputs[nativeRef];
    if (y.isTuple()) {
        auto yTuple = y.toTuple();
        assert (input_size == yTuple -> elements().size());
        for (int i = 0; i < input_size; i++) {
            auto gradTensor = input_tuple[i].toTensor();
            auto outputTensor = yTuple -> elements()[i].toTensor();
            outputTensor.backward(gradTensor);
        }
    } else {
        y.toTensor().backward(input_tuple[0].toTensor());
    }

    // Release critical part
    jenv -> ReleaseIntArrayElements(input_joffset, c_input_offsets, JNI_ABORT);
    for (int i = 0; i < input_size; i++) {
        jenv -> ReleasePrimitiveArrayCritical(j_data_vector[i], c_data_vector[i], 0);
        jenv -> ReleasePrimitiveArrayCritical(j_shape_vector[i], c_shape_vector[i], 0);
    }

    auto modelInput = modelInputs[nativeRef].toTuple();
    int resultSize = modelInput -> elements().size();
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jobjectArray jTensorArray = jenv -> NewObjectArray(resultSize, jtensor_class, NULL);
    for (size_t i = 0; i < resultSize; i++) {
        auto t = modelInput -> elements()[i].toTensor().grad();
        jobject jt = torch2JTensor(jenv, t);
        jenv -> SetObjectArrayElement(jTensorArray, i, jt);
    }
    return jTensorArray;
  }


JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_lossForwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jobjectArray input_jstorage, jintArray input_joffset, jobjectArray input_jshape, jobjectArray label_jstorage, jintArray label_joffset, jobjectArray label_jshape) {

    std::vector<jfloatArray> j_data_vector;
    std::vector<jfloat*> c_data_vector;
    std::vector<jintArray> j_shape_vector;
    std::vector<jint*> c_shape_vector;

    // create input tuple
    int input_size = jenv -> GetArrayLength(input_jstorage);
    jint* c_input_offsets = (jint*) jenv -> GetIntArrayElements(input_joffset, JNI_FALSE);
    std::vector<c10::IValue> input_tuple;

    for (int i = 0; i < input_size; i++) {
        jfloatArray tensor_storage = (jfloatArray)jenv->GetObjectArrayElement(input_jstorage, i);
        jfloat* input_c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(tensor_storage, JNI_FALSE);

        jintArray tensor_shape = (jintArray)jenv->GetObjectArrayElement(input_jshape, i);
        jint* input_c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(tensor_shape, JNI_FALSE);
        int c_dim_count = jenv -> GetArrayLength(tensor_shape);

        std::vector<int64_t> input_torch_shape;
        for (int i = 0; i < c_dim_count; i++) {
            input_torch_shape.push_back(*(input_c_shape + i));
        }

        auto input_tensor = torch::from_blob(input_c_storage + c_input_offsets[i], input_torch_shape, at::kFloat);
        input_tensor.set_requires_grad(true);

        input_tuple.push_back(input_tensor);

        j_data_vector.push_back(tensor_storage);
        c_data_vector.push_back(input_c_storage);
        j_shape_vector.push_back(tensor_shape);
        c_shape_vector.push_back(input_c_shape);
    }
    auto input_table = torch::jit::Tuple::create(input_tuple);

    // create label tuple
    int label_size = jenv -> GetArrayLength(label_jstorage);
    jint* c_label_offsets = (jint*) jenv -> GetIntArrayElements(label_joffset, JNI_FALSE);
    std::vector<c10::IValue> label_tuple;
    for (int i = 0; i < label_size; i++) {
        jfloatArray tensor_storage = (jfloatArray)jenv->GetObjectArrayElement(label_jstorage, i);
        jfloat* label_c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(tensor_storage, JNI_FALSE);

        jintArray tensor_shape = (jintArray)jenv->GetObjectArrayElement(label_jshape, i);
        jint* label_c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(tensor_shape, JNI_FALSE);
        int c_dim_count = jenv -> GetArrayLength(tensor_shape);

        std::vector<int64_t> label_torch_shape;
        for (int i = 0; i < c_dim_count; i++) {
            label_torch_shape.push_back(*(label_c_shape + i));
        }

        auto label_tensor = torch::from_blob(label_c_storage + c_label_offsets[i], label_torch_shape, at::kFloat);
        label_tuple.push_back(label_tensor);

        j_data_vector.push_back(tensor_storage);
        c_data_vector.push_back(label_c_storage);
        j_shape_vector.push_back(tensor_shape);
        c_shape_vector.push_back(label_c_shape);
    }
    auto label_table = torch::jit::Tuple::create(label_tuple);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> lossInputs;
    if (input_table -> elements().size() == 1) {
        lossInputs.push_back(input_table -> elements()[0]);
    } else {
        lossInputs.push_back(input_table);
    }

    if (label_table -> elements().size() == 1) {
        lossInputs.push_back(label_table -> elements()[0]);
    } else {
        lossInputs.push_back(label_table);
    }

    std::shared_ptr<torch::jit::script::Module> loss_ptr = lossHandles[nativeRef];
    assert(loss_ptr != nullptr);
    at::Tensor loss = loss_ptr->forward(lossInputs).toTensor();
    loss.backward();

    std::vector<torch::jit::IValue> grad_tuple;
    for(auto const& value: input_tuple) {
        grad_tuple.push_back(value.toTensor().grad());
    }
    auto grad_table = torch::jit::Tuple::create(grad_tuple);

    mtx.lock();
    lossGrads[nativeRef] = grad_table;
    mtx.unlock();

    // Release critical part
    jenv -> ReleaseIntArrayElements(input_joffset, c_input_offsets, JNI_ABORT);
    jenv -> ReleaseIntArrayElements(label_joffset, c_label_offsets, JNI_ABORT);
    for (int i = 0; i < input_size + label_size; i++) {
        jenv -> ReleasePrimitiveArrayCritical(j_data_vector[i], c_data_vector[i], 0);
        jenv -> ReleasePrimitiveArrayCritical(j_shape_vector[i], c_shape_vector[i], 0);
    }

    jobject result = torch2JTensor(jenv, loss);
    return result;
  }

JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_lossBackwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    auto grad = lossGrads[nativeRef];

    int resultSize = grad.toTuple() -> elements().size();
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");

    jobjectArray jTensorArray = jenv -> NewObjectArray(resultSize, jtensor_class, NULL);
    auto outputTuple = grad.toTuple();
    for (size_t i = 0; i < resultSize; i++) {
        auto t = outputTuple -> elements()[i].toTensor();
        jobject jt = torch2JTensor(jenv, t);
        jenv -> SetObjectArrayElement(jTensorArray, i, jt);
    }
    return jTensorArray;
  }


JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_releaseLossNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    mtx.lock();
    lossHandles.erase(nativeRef);
    lossGrads.erase(nativeRef);
    mtx.unlock();
  }


JNIEXPORT jfloatArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_getGradientNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    std::shared_ptr<torch::jit::script::Module> model_ptr = modelHandles[nativeRef];
    assert(model_ptr != nullptr);
    std::vector<float> gradients;
    getAndZeroGradients(model_ptr, gradients);

    jfloatArray result_storage = jenv -> NewFloatArray(gradients.size());
    jenv -> SetFloatArrayRegion(result_storage, 0, gradients.size(), &gradients[0]);
    return result_storage;
  }


JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_updateWeightNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jfloatArray jstorage) {
    jfloat* c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(jstorage, JNI_FALSE);
    std::shared_ptr<torch::jit::script::Module> model_ptr = modelHandles[nativeRef];
    assert(model_ptr != nullptr);

    int index = 0;
    updateWeights(model_ptr, c_storage, index);

    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    return;
  }


JNIEXPORT jfloatArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_getWeightNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
      std::shared_ptr<torch::jit::script::Module> model_ptr = modelHandles[nativeRef];
      assert(model_ptr != nullptr);

      std::vector<float> weights;
      getWeights(model_ptr, weights);

      jfloatArray result_storage = jenv -> NewFloatArray(weights.size());
      jenv -> SetFloatArrayRegion(result_storage, 0, weights.size(), &weights[0]);
      return result_storage;
  }
}

