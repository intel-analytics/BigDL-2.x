import copy
from io import BytesIO
import numpy as np
from itertools import chain, islice

from enum import Enum
import json


class DType(Enum):
    INT32 = 1
    FLOAT32 = 2


class FeatureType(Enum):
    IMAGE = 1
    NDARRAY = 2
    SCALAR = 3


PUBLIC_ENUMS = {
    "FeatureType": FeatureType,
    "DType": DType
}


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(PUBLIC_ENUMS[name], member)
    else:
        return d


def dict_to_row(schema, row_dict):
    import pyspark
    err_msg = 'Dictionary fields \n{}\n do not match schema fields \n{}'.format(
            '\n'.join(sorted(row_dict.keys())), '\n'.join(schema.keys()))
    assert set(row_dict.keys()) == set(schema.keys()), err_msg

    row = {}
    for k, v in row_dict.items():
        schema_field = schema[k]
        if schema_field["type"] == FeatureType.IMAGE:
            image_path = v
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            row[k] = img_bytes
        elif schema_field["type"] == FeatureType.NDARRAY:
            memfile = BytesIO()
            np.savez_compressed(memfile, arr=v)
            row[k] = bytearray(memfile.getvalue())
        else:
            row[k] = v
    return pyspark.Row(**row)


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))
