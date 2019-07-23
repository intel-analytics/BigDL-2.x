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
std::unordered_map<int, at::Tensor> modelInputs;
std::unordered_map<int, at::Tensor> modelOutputs;
std::unordered_map<int, at::Tensor> lossGrads;
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

jobject torch2JTensor(JNIEnv *jenv, at::Tensor &tensor) {
    // Wrap to Zoo JTensor
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jmethodID jtensor_constructor = jenv -> GetMethodID(jtensor_class, "<init>", "([F[I)V");

    auto sizes = tensor.sizes();

    int result_storage_len = 1;
    float *pytorch_result_storage = tensor.data<float>();
    int result_shape_len = sizes.size();

    int pytorch_result_shape[result_shape_len];
    int j = 0;
    while (j < result_shape_len) {
        pytorch_result_shape[j] = sizes[j];
        result_storage_len *= sizes[j];
        j++;
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
JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_modelForwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jboolean isTraining, jfloatArray jstorage, jint joffset, jintArray jshape) {

    // to Torch Tensor
    jfloat* c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(jstorage, JNI_FALSE);
    jint* c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(jshape, JNI_FALSE);
    int c_shape_len = jenv -> GetArrayLength(jshape);

    //Generate pytorch shape
    std::vector<int64_t> torch_shape;
    int i = 0;
    while(i < c_shape_len) {
        torch_shape.push_back(*(c_shape + i));
        i++;
    }
    // create a Tensor
    auto torch_input_tensor = torch::from_blob(c_storage + joffset, torch_shape, at::kFloat);
    if (isTraining) {
        torch_input_tensor.set_requires_grad(true);
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_input_tensor);

    std::shared_ptr<torch::jit::script::Module> model_ptr = modelHandles[nativeRef];
    assert(model_ptr != nullptr);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = model_ptr->forward(inputs).toTensor();

    if (isTraining) {
        mtx.lock();
        modelInputs[nativeRef] = torch_input_tensor;
        modelOutputs[nativeRef] = output;
        mtx.unlock();
    }

    // Release critical part
    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(jshape, c_shape, 0);

    jobject result = torch2JTensor(jenv, output);
    return result;
  }


JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_releaseModelNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    mtx.lock();
    modelHandles.erase(nativeRef);
    modelInputs.erase(nativeRef);
    modelOutputs.erase(nativeRef);
    mtx.unlock();
  }

JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_modelBackwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jfloatArray jstorage, jint joffset, jintArray jshape) {

    // to Torch Tensor
    jfloat* c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(jstorage, JNI_FALSE);
    jint* c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(jshape, JNI_FALSE);
    int c_shape_len = jenv -> GetArrayLength(jshape);

    //Generate pytorch shape
    std::vector<int64_t> torch_shape;
    int i = 0;
    while(i < c_shape_len) {
        torch_shape.push_back(*(c_shape + i));
        i++;
    }
    // create gradOutput Tensor
    auto gradOutput_tensor = torch::from_blob(c_storage + joffset, torch_shape, at::kFloat);

    at::Tensor y = modelOutputs[nativeRef];
    y.backward(gradOutput_tensor);

    auto gradInput = modelInputs[nativeRef].grad();

    // Release critical part
    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(jshape, c_shape, 0);

    jobject result = torch2JTensor(jenv, gradInput);
    return result;
  }


JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_lossForwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jfloatArray input_jstorage, jint input_joffset, jintArray input_jshape, jfloatArray label_jstorage, jint label_joffset, jintArray label_jshape) {

    // create input Tensor
    jfloat* input_c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(input_jstorage, JNI_FALSE);
    jint* input_c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(input_jshape, JNI_FALSE);
    int input_c_shape_len = jenv -> GetArrayLength(input_jshape);

    std::vector<int64_t> input_torch_shape;
    int i = 0;
    while(i < input_c_shape_len) {
        input_torch_shape.push_back(*(input_c_shape + i));
        i++;
    }

    auto input_tensor = torch::from_blob(input_c_storage + input_joffset, input_torch_shape, at::kFloat);
    input_tensor.set_requires_grad(true);

    // create label Tensor
    jfloat* label_c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(label_jstorage, JNI_FALSE);
    jint* label_c_shape = (jint*) jenv -> GetPrimitiveArrayCritical(label_jshape, JNI_FALSE);
    int label_c_shape_len = jenv -> GetArrayLength(label_jshape);

    std::vector<int64_t> label_torch_shape;
    i = 0;
    while(i < label_c_shape_len) {
        label_torch_shape.push_back(*(label_c_shape + i));
        i++;
    }
    auto label_tensor = torch::from_blob(label_c_storage + label_joffset, label_torch_shape, at::kFloat);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> lossInputs;
    lossInputs.push_back(input_tensor);
    lossInputs.push_back(label_tensor);

    std::shared_ptr<torch::jit::script::Module> loss_ptr = lossHandles[nativeRef];
    assert(loss_ptr != nullptr);
    at::Tensor loss = loss_ptr->forward(lossInputs).toTensor();
    loss.backward();

    mtx.lock();
    lossGrads[nativeRef] = input_tensor.grad();
    mtx.unlock();

    // Release critical part
    jenv -> ReleasePrimitiveArrayCritical(input_jstorage, input_c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(input_jshape, input_c_shape, 0);
    jenv -> ReleasePrimitiveArrayCritical(label_jstorage, label_c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(label_jshape, label_c_shape, 0);

    jobject result = torch2JTensor(jenv, loss);
    return result;
  }

JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_lossBackwardNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    auto grad = lossGrads[nativeRef];

    jobject result = torch2JTensor(jenv, grad);
    return result;
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

