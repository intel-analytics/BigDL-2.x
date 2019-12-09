# API Guide

## Python 

### Input
The class `Input` defines methods allowing you to input data into Cluster Serving [Input Pipeline]().

#### __init__

[view source]()

```
__init__()
```
sets up a connection with configuration in `config_path`
#### enqueue_image

[view source]()

```
enqueue_image()
```
puts `data` with identification `id` into Pipeline with `.jpg` encoding.

### Output
The class `Output` defines methods allowing you to get result from Cluster Serving [Output Pipeline]().

#### __init__

[view source]()

```
__init__()
```
sets up a connection with configuration in `config_path`

#### get_result
