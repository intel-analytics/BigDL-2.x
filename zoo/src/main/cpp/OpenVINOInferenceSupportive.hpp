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

#include <inference_engine/inference_engine.hpp>
#include "CTensor.hpp"

using namespace InferenceEngine;

enum OpenVINODeviceName
{
    CPU = 0,
    GPU = 1
};

class OpenVINOInferenceSupportive
{
  public:
    explicit OpenVINOInferenceSupportive();

    static ExecutableNetwork* loadOpenVINOIR(const std::string modelFilePath, const std::string weightFilePath, const int deviceType);
 
    static CTensor<float> predict(ExecutableNetwork executable_network, CTensor<float> datatensor);

    static void destoryExecutableNetworkPtr(ExecutableNetwork * pexe) { free(pexe); }
  private:

};

