#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.0.0)

set (CMAKE_CXX_FLAGS "-std=c++11")
set (LIB_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../../../target/classes")
set (INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../target/openvino/include")
set (InferenceEngine_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../target/openvino/include/inference_engine")
set (InferenceEngine_LIBRARIES inference_engine)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../target/classes)

message (${INCLUDE_DIRS})

find_package(JNI)
if(JNI_FOUND)
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message(STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()

include_directories (
    ${InferenceEngine_INCLUDE_DIRS}
    ${INCLUDE_DIRS}
    ${JNI_INCLUDE_DIRS}
)

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})

link_directories(
    ${LIB_FOLDER}
    ${JNI_LIBRARIES}
)

# Create library file from sources.
# add_library(inference_engine_interface SHARED ${MAIN_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/include/extension/ext_list.cpp)

add_library(zoo_inference SHARED ${MAIN_SRC})

target_link_libraries(zoo_inference inference_engine cpu_extension dl pthread)

