1. Download VOC2007 and VOC2012 dataset.
By default, we assume the ssd project root is ```ssd_root=${HOME}/analytics-zoo/pipeline/objectDetection```,
and data is stored in ```data_root=${ssd_root}/data/pascal```

You may want to modify ssd_root as needed.

```bash
./data/pascal/get_pascal.sh
```

It should have this basic structure

```
$data_root/VOCdevkit/                           # development kit
$data_root/VOCdevkit/VOC2007                    # VOC2007 image sets, annotations, etc.
$data_root/VOCdevkit/VOC2012                    # VOC2012 image sets, annotations, etc.
```

2. Convert to Sequence Files
```bash
./data/pascal/convert_pascal.sh
```

Put sequence file to hdfs
```bash
hdfs dfs -put ${data_root}/seq/ hdfs://xxx/xxx
```