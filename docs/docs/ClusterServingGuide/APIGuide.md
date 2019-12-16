# API Guide

## Python 

### Input
The class `Input` defines methods allowing you to input data into Cluster Serving [Input Pipeline]().

#### __init__

[view source]()

```
__init__(config_path)
```
sets up a connection with configuration in `config_path`

_return_: None

`config_path`: the file path of your Cluster Serving [configuration file]() `config.yaml`.
#### enqueue_image
[view source]()

```
enqueue_image(uri, img)
```
puts image `img` with identification `uri` into Pipeline with JPG encoding.

_return_: None

`uri`: a string, unique identification of your image

`img`: `ndarray` of your image, could be loaded by `cv2.imread()` of opencv-python package.

_Example_
```
from zoo.serving.client.helpers import Input
import cv2
input_api = Input()
img = cv2.imread(/path/to/image)
input_api.enqueue_image("my-image", img)
```

### Output
The class `Output` defines methods allowing you to get result from Cluster Serving [Output Pipeline]().
#### __init__
[view source]()

```
__init__(config_path)
```
sets up a connection with configuration in `config_path`

`config_path`: the file path of your Cluster Serving [configuration file]() `config.yaml`.
#### query
[view source]()

```
query(uri)
```
query result in output Pipeline by key `uri`

_return_: dict(), string type, the output of your prediction, which can be parsed by json.

_Example_
```
from zoo.serving.client.helpers import Output
import json
output_api = Output()
d = output_api.query("my-image") 
json.loads(d)
```

#### dequeue
[view source]()

```
get_result()
```
gets all result of your model prediction and dequeue them from Pipeline

_return_: dict(), with keys the `uri` of your [enqueue], string type, and values the output of your prediction, string type, which can be parsed by json.

_Example_
```
from zoo.serving.client.helpers import Output
import json
output_api = Output()
d = output_api.dequeue()
for k in d.keys():
  class_prob_map = json.loads(d[k])
```



