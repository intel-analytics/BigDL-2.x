1. Download Images and Annotations from MSCOCO.

By default, we assume the ssd project root is
```ssd_root=${HOME}/analytics-zoo/pipeline/ssd```,
and data is stored in ```data_root=${ssd_root}/data/coco```

You may want to modify ssd_root as needed.

```bash
./data/coco/get_coco.sh
```

It should have this basic structure

```
-$data_root/images/                        # images
$data_root/annotations/                    # annotations
```

2. Parse coco annotations
```bash
git clone https://github.com/weiliu89/coco.git
cd coco
git checkout dev
cd PythonAPI
python setup.py build_ext --inplace
# Check scripts/batch_split_annotation.py and change settings accordingly.
python scripts/batch_split_annotation.py
```

3. Create ImageSet
```bash
python data/coco/create_list.py
```

4. Convert to Sequence Files
```bash
./data/coco/convert_pascal.sh
```

Put sequence file to hdfs
```bash
hdfs dfs -put ${data_root}/seq/ hdfs://xxx/xxx
```