from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Row

def sql_dataframe_example(spark):
    
    print("INFO SQL Dataframe Example starts")
    
    df = spark.read.parquet("work/spark-2.4.6/examples/src/main/resources/users.parquet")
    df.show()
    
    # agg
    res = df.agg({"name": "max"}).collect()
    print(res)
    print("agg API finished")

    # alias
    df_as1 = df.alias("df_as1")
    df_as2 = df.alias("df_as2")
    joined_df = df_as1.join(df_as2, col("df_as1.name") == col("df_as2.name"), 'inner')
    res = joined_df.select("df_as1.name", "df_as2.name", "df_as2.favorite_color").collect()
    print(res)
    print("alias API finished")

    # colRegex, collect and columns
    df = spark.createDataFrame([("a", 1.1, 1.0), ("b", 2.2, 2.0), ("c", 3.3, 3.0)], ["Col1", "Col2", "Col3"])
    df.select(df.colRegex("`(Col1)?+.+`")).show()
    print(df.collect())
    print(df.columns)
    print("colRegex, collect and columns API finished")

    # approxQuantile
    print(df.approxQuantile("Col2", [0.1], 1.0))
    print("approxQuantile finished")
    
    # cache and checkpoint
    df.cache()
    # print(df.checkpoint())
    print("cache API finished")

    # coalesce
    res = df.coalesce(1).rdd.getNumPartitions()
    print(res)
    print("coalesce API finished")

    # corr, count and cov
    res = df.corr("Col2", "Col3")
    print(res) 
    res = df.count()
    print(res)
    res = df.cov("Col2", "Col3")
    print(res)
    print("corr, count, cov API finished")
    

    # GlobalTempView
    df = spark.createDataFrame([("a", 1.1, 1.0), ("b", 2.2, 2.0), ("c", 3.3, 3.0)], ["Col1", "Col2", "Col3"])
    df.createGlobalTempView("threecols")
    df2 = spark.sql("select * from global_temp.threecols")
    print(sorted(df.collect()) == sorted(df2.collect()))
    df2 = df.filter(df.Col2 > 2.0)
    df2.createOrReplaceGlobalTempView("threecols")
    df3 = spark.sql("select * from global_temp.threecols")
    print(sorted(df3.collect()) == sorted(df2.collect()))
    print("GlobalTempView API finished")

    # TempView
    df.createTempView("threecols")
    df2 = spark.sql("select * from threecols")
    print(sorted(df.collect()) == sorted(df2.collect()))
    df2 = df.filter(df.Col2 > 2.0)
    df2.createOrReplaceTempView("threecols")
    df3 = spark.sql("select * from threecols")
    print(sorted(df3.collect()) == sorted(df2.collect()))
    print("TempView API finished")

    # cross
    df = spark.createDataFrame([("a", 1.1, 1.0), ("b", 2.2, 2.0), ("c", 3.3, 3.0)], ["Col1", "Col2", "Col3"])
    df2 = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    df3 = spark.createDataFrame([("Tom", 80), ("Bob", 85)], ["name", "height"])
    df2.crossJoin(df3.select("height")).select("age", "name", "height").show()
    res = df.crosstab("Col1", "Col2")
    print(res)
    print("cross API finished")

    # cube
    df2.cube("name", df2.age).count().orderBy("name", "age").show()
    print("cube API finshed")

    # descibe
    df2.describe().show()
    print("descibe API finshed")

    # distinct
    res = df.distinct().count()
    print(res)
    print("distinct API finshed")

    # drop
    df3 = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    df3.dropDuplicates().show()
    df3.drop('age').show()
    df3.na.drop().show()
    print(df3.dtypes)
    print("drop API finshed")

    # exceptAll, explain, fill and first
    df1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
    df2 = spark.createDataFrame([("a", 1), ("b", 3)], ["C1", "C2"])
    df1.exceptAll(df2).show()
    df1.explain()
    df1.na.fill(50).show()
    res = df1.first()
    print(res)
    print("exceptAll, explain, fill and first API finshed")

    # foreach
    df1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
    df2 = spark.createDataFrame([("a", 1), ("b", 3)], ["C1", "C2"])
    def f(row):
        print(row.C1)
    df1.foreach(f)
    print("foreach API finished")
    def F(table):
        for row in table:
            print(row.C1)
    df1.foreachPartition(F)
    print("foreachPartition API finshed") 

    # freqItems and groupBy
    df1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
    res = df1.freqItems(["C1"])
    print(res)
    res = df1.groupBy().avg().collect()
    print(res)
    print("freqItems and groupBy API finshed")
    
    # head
    res = df1.head(1)
    print(res)
    print("head API finshed")

    # intersect and join
    df1 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 4)], ["C1", "C2"])
    df2 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 0)], ["C1", "C2"])
    df1.join(df2.hint("broadcast"), "C1").show()
    df1.intersect(df2).sort("C1", "C2").show()
    df1.intersectAll(df2).sort("C1", "C2").show()
    print("intersect and join API finshed")

    # isLocal and isStreaming
    res = df1.isLocal()
    print(res)
    print(df1.isStreaming)
    res = df1.limit(2).collect()
    print(res)
    print("isLocal and isStreaming API finshed")

    # orderBy
    df2 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 0)], ["C1", "C2"])
    df2.orderBy(df2.C2.desc()).show()
    df2.persist()
    df2.unpersist()
    df2.printSchema()
    print("orderBy API finshed")

    # randomSplit
    df4 = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    splits = df4.randomSplit([1.0, 2.0], 24)
    splits[0].show()

    df4.registerTempTable("people")
    df2 = spark.sql("select * from people")
    print(sorted(df4.collect()) == sorted(df2.collect()))
    spark.catalog.dropTempView("people")
    print("randomSplit API finshed")
    
    # repartition
    df4 = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    data = df4.union(df4).repartition("age")
    data.show()
    data = df4.repartitionByRange("age")
    data.show()
    print("repartition API finshed")
    
    # replace
    data = df4.na.replace('Alice', None)
    data.show()
    df4.rollup("name", df4.age).count().orderBy("name", "age").show()
    print("replace API finshed")

    # Sample and sampleBy
    df = spark.range(10)
    res = df.sample(0.5, 3).count()
    print(res)
    res = df.sample(fraction=0.5, seed=3).count()
    print(res)
    dataset = spark.range(0, 100).select((col("id") % 3).alias("key"))
    sampled = dataset.sampleBy("key", fractions={0: 0.1, 1: 0.2}, seed=0)
    sampled.groupBy("key").count().orderBy("key").show()
    print(df.schema)
    print("Sample and sampleBy API finshed")
    
    # select
    df = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    res = df.select(df.name, (df.age + 10).alias('age')).collect()
    print(res)
    res = df.selectExpr("age * 2", "abs(age)").collect()
    print(res)
    df.sortWithinPartitions("age", ascending=False).show()
    res = df.storageLevel
    print(res)
    print("select API finshed")

    # substract, summary and take 
    df = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    df2 = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80)], ["name", "age", "height"])
    df.subtract(df2)
    df.summary().show()
    res = df.take(2)
    print(res)
    print("substract, summary and take API finshed")

    # to
    df = spark.createDataFrame([('Alice', 5), ('Alice', 5)], ["name", "age"])
    res = df.toDF('f1', 'f2').collect()
    print(res)
    res = df.toJSON().first()
    print(res)
    res = list(df.toLocalIterator())
    print(res)
    res = df.toPandas()
    print(res)
    print("to API finshed") 

    # union
    df = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80), ('Alice', 10, 80)], ["name", "age", "height"])
    df2 = spark.createDataFrame([('Alice', 5, 80), ('Alice', 5, 80)], ["name", "age", "height"])
    df.union(df2).show()
    df.unionAll(df2).show()

    df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
    df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col0"])
    df1.unionByName(df2).show()
    print("union API finshed") 

    # with
    df = spark.createDataFrame([('Alice', 5), ('Alice', 5)], ["name", "age"]) 
    res = df.withColumn('age2', df.age + 2).collect()
    print(res)
    res = df.withColumnRenamed('age', 'age2').collect()
    print(res)
    # res = df.select('name', df.time.cast('timestamp')).withWatermark('time', '10 minutes')
    # print(res)
    print("with API finshed")

    # write
    res = df.write
    print(res)
    # res = df.writeStream
    # print(res)
    print("write API finshed")

    print("INFO SQL Dataframe Example finishes")

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
    .appName("Python Spark SQL Dataframe example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    sql_dataframe_example(spark)
