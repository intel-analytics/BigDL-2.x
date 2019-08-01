# OpenVINO

## Requirements
1. Python & TensorFlow
2. OpenVINO

## Auto Calibration
Assume you have OpenVINO IR model (converted from Caffe or TensorFlow), and you want to convert it into `int8` model. Note that OpenVINO `int8` IR promises higher performance than IR model (FP32) by sacrificing a few precision (1% by default).

To achive that, you can use OpenVINO `calibration_tool` or use our `auto_calibration`, which provides simpler API based on `calibration_tool`.

Basic Usage
```bash
python auto_calibration.py -m {OpenVINO IR model, *.xml} -i {validation data}
```

API document
```bash
python auto_calibration.py -h
```

**Validation Data**
1. Image Classification



2. Object Detection






## References
1. [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) 
2. [Analytics-Zoo](https://github.com/intel-analytics/analytics-zoo)
3. [TensorFlow](https://www.tensorflow.org/)

