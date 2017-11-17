CREATE EXTERNAL TABLE
 ${hiveconf:table}(uri string, bytes binary)
 STORED BY 'org.apache.hadoop.hive.hbase.HBaseStorageHandler'
 WITH SERDEPROPERTIES ("hbase.columns.mapping" =
 ":key,bytes:bytes")