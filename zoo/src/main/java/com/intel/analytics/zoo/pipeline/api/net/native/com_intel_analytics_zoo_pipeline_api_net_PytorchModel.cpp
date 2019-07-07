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
std::unordered_map<int, std::shared_ptr<torch::jit::script::Module>> handles;
std::unordered_map<int, std::shared_ptr<torch::jit::script::Module>> lossHandles;
std::unordered_map<int, at::Tensor> outputs;
long modelID;


auto getWeights(std::shared_ptr<torch::jit::script::Module> m, std::vector<float> &xv) -> int {
    auto children = m->get_modules();

    if (children.size() == 0) {
        auto slots = m -> get_parameters();
        std::cout << "slot size: " << slots.size() << std::endl;
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

auto getGradients(std::shared_ptr<torch::jit::script::Module> m, std::vector<float> &xv) -> int {
    auto children = m->get_modules();

    if (children.size() == 0) {
        auto slots = m -> get_parameters();

        for (size_t i = 0; i < slots.size(); ++i) {
            auto& x = slots[i];
            if (x.value().toTensor().grad().defined()) {
                size_t x_size = x.value().toTensor().grad().numel();
                auto p = static_cast<float*>(x.value().toTensor().grad().storage().data());
                for(size_t i = 0; i < x_size; i++)
                {
                    xv.push_back(p[i]);
                }
            }
        }
    } else {
        for (const auto& child : children) {
            getGradients(child, xv);
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


JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_loadNative
  (JNIEnv *jenv, jclass jobj, jstring jmodel_path, jstring jloss_path) {
    const char* p_model_path = jenv->GetStringUTFChars(jmodel_path, NULL);
    const char* p_loss_path = jenv->GetStringUTFChars(jloss_path, NULL);

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> model_ptr = torch::jit::load(p_model_path);
    std::shared_ptr<torch::jit::script::Module> loss_ptr = torch::jit::load(p_loss_path);
    assert(model_ptr != nullptr);
    assert(loss_ptr != nullptr);

    mtx.lock();
    modelID++;
    long id = modelID;
    handles[id] = model_ptr;
    lossHandles[id] = loss_ptr;
    mtx.unlock();

    jenv->ReleaseStringUTFChars(jmodel_path, p_model_path);
    jenv->ReleaseStringUTFChars(jloss_path, p_loss_path);
    return id;
  }


/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    forward
 * Signature: ([F[I)Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_forwardNative
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
    // create a Tensor
    auto torch_input_tensor = torch::from_blob(c_storage + joffset, torch_shape, at::kFloat);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_input_tensor);

    std::shared_ptr<torch::jit::script::Module> model_ptr = handles[nativeRef];

    // Execute the model and turn its output into a tensor.
    at::Tensor output = model_ptr->forward(inputs).toTensor();
    mtx.lock();
    outputs[nativeRef] = output;
    mtx.unlock();

    // Release critical part
    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(jshape, c_shape, 0);

    // Wrap to Zoo JTensor
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jmethodID jtensor_constructor = jenv -> GetMethodID(jtensor_class, "<init>", "([F[I)V");

    auto sizes = output.sizes();

    int result_storage_len = 1;
    float *pytorch_result_storage = output.data<float>();
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

    jobject result = jenv -> NewObject(jtensor_class, jtensor_constructor, result_storage, result_shape);

    return result;
  }


JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_releaseNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    mtx.lock();
    handles.erase(nativeRef);
    mtx.unlock();
  }

JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_backwardNative
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
    // create a Tensor
    auto label_tensor = torch::from_blob(c_storage + joffset, torch_shape, at::kFloat);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> lossInputs;

    at::Tensor y = outputs[nativeRef];
    lossInputs.push_back(y);
    lossInputs.push_back(label_tensor);

    std::shared_ptr<torch::jit::script::Module> loss_ptr = lossHandles[nativeRef];
    std::cout << "lossInputs is: " << std::endl;
    std::cout << y << std::endl;
    std::cout << label_tensor << std::endl;
    at::Tensor loss = loss_ptr->forward(lossInputs).toTensor();
    std::cout << "loss is: " << std::endl;
    std::cout << loss << std::endl;
    loss.backward();

    // Release critical part
    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    jenv -> ReleasePrimitiveArrayCritical(jshape, c_shape, 0);

    // Wrap to Zoo JTensor
    jclass jtensor_class = jenv -> FindClass("com/intel/analytics/zoo/pipeline/inference/JTensor");
    jmethodID jtensor_constructor = jenv -> GetMethodID(jtensor_class, "<init>", "([F[I)V");

    auto sizes = loss.sizes();

    int result_storage_len = 1;
    float *pytorch_result_storage = loss.data<float>();
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

    jobject lossJTensor = jenv -> NewObject(jtensor_class, jtensor_constructor, result_storage, result_shape);

    return lossJTensor;
  }


JNIEXPORT jfloatArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_getGradientNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
    std::shared_ptr<torch::jit::script::Module> model_ptr = handles[nativeRef];
    std::vector<float> gradients;
    getGradients(model_ptr, gradients);

    jfloatArray result_storage = jenv -> NewFloatArray(gradients.size());
    jenv -> SetFloatArrayRegion(result_storage, 0, gradients.size(), &gradients[0]);
    return result_storage;
  }


JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_updateWeightNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef, jfloatArray jstorage) {
    jfloat* c_storage = (jfloat*) jenv -> GetPrimitiveArrayCritical(jstorage, JNI_FALSE);
    std::shared_ptr<torch::jit::script::Module> model_ptr = handles[nativeRef];

    int index = 0;
    updateWeights(model_ptr, c_storage, index);

    jenv -> ReleasePrimitiveArrayCritical(jstorage, c_storage, 0);
    return;
  }


JNIEXPORT jfloatArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_getWeightNative
  (JNIEnv * jenv, jclass jobj, jlong nativeRef) {
      std::shared_ptr<torch::jit::script::Module> model_ptr = handles[nativeRef];
      std::vector<float> weights;
      getWeights(model_ptr, weights);

      jfloatArray result_storage = jenv -> NewFloatArray(weights.size());
      jenv -> SetFloatArrayRegion(result_storage, 0, weights.size(), &weights[0]);
      return result_storage;
  }
}


JNIEXPORT int JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_test
  (JNIEnv * jenv, jclass jobj){

    auto p_model_path = "/home/yuhao/PycharmProjects/pytorch_test/pts/model_221.pt";
    std::shared_ptr<torch::jit::script::Module> model_ptr = torch::jit::load(p_model_path);

    auto p_loss_path = "/home/yuhao/PycharmProjects/pytorch_test/pts/loss.pt";
    std::shared_ptr<torch::jit::script::Module> loss_ptr = torch::jit::load(p_loss_path);

    for (int ii = 0; ii < 5; ii++) {
        std::cout << "\n------------------------iteration: " << ii <<  "\n";

        std::vector<torch::jit::IValue> modelInputs;
        modelInputs.push_back(torch::ones({2, 2}));

        auto output = model_ptr->forward(modelInputs).toTensor();

        std::vector<torch::jit::IValue> lossInputs;
        lossInputs.push_back(output);
        lossInputs.push_back(torch::ones({2, 1}));

        auto loss = loss_ptr->forward(lossInputs).toTensor();
        std::cout << "\nbackwarding: \n";
        loss.backward();

        std::cout << "\nupdate weights: \n";
        int index = 0;
        float arr4[9] = { 1, 2 };
        updateWeights(model_ptr, arr4, index);

    }
    return 1;
}