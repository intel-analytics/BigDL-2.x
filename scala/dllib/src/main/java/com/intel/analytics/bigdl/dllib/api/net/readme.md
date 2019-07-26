This is a doc for generating the PytorchModel native libraries. Tested on: torch 1.1.0
 and torchvision 0.2.2

# Download LibTorch
Download LibTorch zip file, unzip it and check the build-version file (1.1.0)

# Build dynamic lib
```bash
cd path-to-analytics-zoo/zoo/target/classes
javah -d path-to-analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/pipeline/api/net/native  com.intel.analytics.zoo.pipeline.api.net.PytorchModel
```

Update the path in CMakeLists.txt to add your JDK path
```bash
cd path-to-analytics-zoo/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/pipeline/api/net/native/build
cmake -DCMAKE_PREFIX_PATH=unzipped-libtorch-folder ..
make
```

# Add to resources
add the generated libpytorch-engine.so in build folder to /zoo/src/main/resources/pytorch/lib.zip, thus
lib.zip contains all files in `libtorch/lib/` and `libpytorch-engine.so` from zoo.

