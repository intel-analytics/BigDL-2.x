# RNN based recommendation service using BigDL, Mleap, and Play
A example on how to serve rnn recommendation model trained by BigDL and Spark using BigDL, Mleap, and Play

## Requirements
```scala
scalaVersion := "2.11.12"

val jacksonVersion = "2.6.7"
val sparkVersion = "2.3.1"
val analyticsZooVersion = "0.3.0"

libraryDependencies += guice
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.7.1-spark_2.3.1" % analyticsZooVersion
libraryDependencies += "com.fasterxml.jackson.module" %% "jackson-module-scala" % jacksonVersion
libraryDependencies += "com.amazonaws" % "aws-java-sdk" % "1.11.354"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % "0.12.0"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark-extension" % "0.12.0"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)

dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % jacksonVersion
dependencyOverrides += "com.google.guava" % "guava" % "17.0"
```

## How to run
Using Intellij IDE
```scala
sbt runProd
```
Package code to one jar and run directly using the java command
```scala
sbt -J-Xmx2G assembly
java -Xmx1g -jar ${location of assembled jar}
```

## API Introduction

### Check health
**Path:** /`healthcheck`  
**Methods:** GET  
**Params:**  
None  
**Return:**
```json
{"status": "ok"}
```

### Check model version
**Path:** /`versioncheck`  
**Methods:** GET  
**Params:**  
None  
**Return:**
```json
{
    "status": "ok",
    "recModelVersion": "2019-01-04 13:22:01.000",
    "skuIndexerModelVersion": "2019-01-04 12:49:02.000"
}
```

### Run recommendation model
**Path:** /`recModel`  
**Methods:** POST  
**Params:**  
```json
{
    "SESSION_ID": "12345",
    "SKU_NUM": ["798644", "799238", "8284111"]
}
``` 
**Return:**
```json
[
    {
        "predict1": "9917369",
        "probability1": 1.529609283755619e-8
    },
    {
        "predict2": "6979858",
        "probability2": 1.029998573353202e-9
    },
    {
        "predict3": "604732",
        "probability3": 0.000009014722038252767
    },
    {
        "predict4": "2774708",
        "probability4": 1.881455143683354e-8
    },
    {
        "predict5": "6630902",
        "probability5": 0.000024642282465106056
    },
    {
        "predict6": "4474320",
        "probability6": 1.516945401806127e-7
    },
    {
        "predict7": "776948",
        "probability7": 5.799234746833089e-7
    },
    {
        "predict8": "299388",
        "probability8": 5.195704205386801e-7
    },
    {
        "predict9": "6979871",
        "probability9": 0.000002128666725435638
    },
    {
        "predict10": "147954",
        "probability10": 0.000014428748498349116
    }
]
```  
**Error_message:** ""Nah nah nah nah nah...this request contains bad characters...""

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * Luyang, Wang (tmacraft@hotmail.com)


