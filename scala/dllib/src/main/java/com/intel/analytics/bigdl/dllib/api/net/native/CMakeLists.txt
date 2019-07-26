cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pytorch-engine)

find_package(Torch REQUIRED)

include_directories(/home/yuhao/programs/jdk1.8.0_121/include
                   /home/yuhao/programs/jdk1.8.0_121//include/linux
                   )

add_library(pytorch-engine SHARED
com_intel_analytics_zoo_pipeline_api_net_PytorchModel.cpp
)

set(CMAKE_EXE_LINKER_FLAGS " -static")

target_link_libraries(pytorch-engine  ${TORCH_LIBRARIES} -static-libgcc -static-libstdc++)

install(TARGETS pytorch-engine DESTINATION lib)
install(FILES com_intel_analytics_zoo_pipeline_api_net_PytorchModel.h  DESTINATION include)
set_property(TARGET pytorch-engine PROPERTY CXX_STANDARD 11)

