import numpy as np
from zoo.orca import init_orca_context
from zoo.friesian.feature import FeatureTable
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, FloatType

sc = init_orca_context(cores="*")

LABEL_COL = "_c0"
INT_COLS = ["_c%d" % i for i in list(range(1, 14))]
CAT_COLS = ["_c%d" % i for i in list(range(14, 40))]

tbl = FeatureTable.read_csv("/home/kai/RS/reco_dummy_pipeline/data/small.csv")
tbl = tbl.cast([LABEL_COL] + INT_COLS, "int")
tbl.show(5)
count = tbl.size()
print("Total number of records", count)

tbl = tbl.fillna(np.iinfo(np.int32).min, INT_COLS)
tbl = tbl.fillna("80000000", CAT_COLS)
tbl.show(5)

tbl = tbl.normalize(INT_COLS)
tbl.show(5)

tbl, idx_list = tbl.category_encode(CAT_COLS)
tbl.show(5)

for idx in idx_list:
    size = idx.size()
    print(size)
    if size < 100:
        tbl = tbl.one_hot_encode(idx.col_name, sizes=size)
tbl.show(5)

train_tbl, test_tbl = tbl.split([0.999, 0.001])
print("Train table")
train_tbl.show(5)
print("Test table")
test_tbl.show(5)
print("Finished")
