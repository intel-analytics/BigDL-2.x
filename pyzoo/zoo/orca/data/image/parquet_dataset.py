import json

from pyspark import SparkContext
from pyspark.sql import SparkSession

from zoo.orca.data import XShards, SparkXShards
from zoo.orca.data.file import open_text, write_text
from zoo.orca.data.image.utils import chunks, dict_to_row, EnumEncoder, as_enum, row_to_dict, encode_schema, \
    decode_schema
from bigdl.util.common import get_node_and_core_number
import os


class ParquetDataset:

    @staticmethod
    def write(path, generator, schema, block_size=1000, write_mode="overwrite", **kwargs):
        """
        Take each record in the generator and write it to a parquet file.
        
        **generator**
        Each record in the generator is a dict, the key is a string and will be the column name of saved parquet record and
        the value is the data.
        
        (
        Other options:
        
        1. generator output a two-element tuple representing feature and label. feature and label are both lists contain
           the actual data. The columns in the saved parquet file will be "feature_col_1", "feature_col_2", ..., "label_col_1", "label_col_2"
        2. enforce feature number to be one and it must be a image.
        )
        
        **schema**
        schema defines the name, dtype, shape of a column, as well as the feature type of a column. The feature type, defines
        how to encode and decode the column value.
        
        There are three kinds of feature type:
        1. Scalar, such as a int or float number, or a string, which can be directly mapped to a parquet type
        2. NDarray, which takes a np.ndarray and save it serialized bytes. The corresponding parquet type is BYTE_ARRAY .
        3. Image, which takes a string representing a image file in local file system and save the raw file content bytes.
           The corresponding parquet type is BYTE_ARRAY.
           
        (
        Other options:
        1. enforce all types to be ndarray. For images, it may takes much larger space to save ndarray then image file bytes. 
        2. enforce feature to a image, and labels are ndarrays.  
        )
           
        We can save the schema info to file in the output path.
        
        (
        Other options:
        1. infer from data when reading?
        2. save schema info as a column?
        )
        
        
        :param path: the output path, e.g. file:///output/path, hdfs:///output/path
        :param generator: generate a dict, whose key is a string and value is one of (a scalar value, ndarray, image file path)
        :param schema: a dict, whose key is a string, value is one of (schema_field.Scalar, schema_field.NDarray, schema_field.Image)
        :param kwargs: other args
        """

        sc = SparkContext.getOrCreate()
        spark = SparkSession(sc)
        node_num, core_num = get_node_and_core_number()
        for i, chunk in enumerate(chunks(generator, block_size)):
            chunk_path = os.path.join(path, f"chunk={i}")
            rows_rdd = sc.parallelize(chunk, core_num * node_num).map(lambda x: dict_to_row(schema, x))
            spark.createDataFrame(rows_rdd).write.mode(write_mode).parquet(chunk_path)
        metadata_path = os.path.join(path, "_orca_metadata")

        write_text(metadata_path, encode_schema(schema))

    @staticmethod
    def _read_as_dict_rdd(path):
        sc = SparkContext.getOrCreate()
        spark = SparkSession(sc)
        df = spark.read.parquet(path)
        schema_path = os.path.join(path, "_orca_metadata")

        j_str = open_text(schema_path)[0]

        schema = decode_schema(j_str)

        rdd = df.rdd.map(lambda r: row_to_dict(schema, r))
        return rdd, schema

    @staticmethod
    def _read_as_xshards(path):
        rdd, schema = ParquetDataset._read_as_dict_rdd(path)

        def merge_records(schema, iter):

            l = list(iter)
            result = {}
            for k in schema.keys():
                result[k] = []
            for i, rec in enumerate(l):

                for k in schema.keys():
                    result[k].append(rec[k])
            return [result]

        result_rdd = rdd.mapPartitions(lambda iter: merge_records(schema, iter))
        xshards = SparkXShards(result_rdd)
        return xshards


    @staticmethod
    def read_as_tf(path):
        """
        return a orca.data.tf.data.Dataset
        :param path: 
        :return: 
        """
        from zoo.orca.data.tf.data import Dataset
        xshards = ParquetDataset._read_as_xshards(path)
        return Dataset.from_tensor_slices(xshards)

    @staticmethod
    def read_as_torch(path):
        """
        return a orca.data.torch.data.DataLoader
        :param path: 
        :return: 
        """

    @staticmethod
    def write_imagenet(input_path, output_path, **kwargs):
        pass

    @staticmethod
    def write_coco(input_path, output_path, **kwargs):
        pass

    @staticmethod
    def write_voc(input_path, output_path, **kwargs):
        pass

    @staticmethod
    def write_mnist(images, labels, output_path):
        pass

    @staticmethod
    def write_fashion_mnist(images, labels, output_path):
        pass
