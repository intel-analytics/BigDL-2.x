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
{
    "0": {
        "heading": "Office Depot top sellers",
        "name": "content2_json",
        "sku": [
            {
                "1": {
                    "id": "9917369",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "2": {
                    "id": "6979858",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "3": {
                    "id": "604732",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "4": {
                    "id": "2774708",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "5": {
                    "id": "6630902",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "6": {
                    "id": "4474320",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "7": {
                    "id": "776948",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "8": {
                    "id": "299388",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "9": {
                    "id": "6979871",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "10": {
                    "id": "147954",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "11": {
                    "id": "282127",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "12": {
                    "id": "178878",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "13": {
                    "id": "485681",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "14": {
                    "id": "351975",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "15": {
                    "id": "460958",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "16": {
                    "id": "470796",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "17": {
                    "id": "465617",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "18": {
                    "id": "493894",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "19": {
                    "id": "4981958",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            },
            {
                "20": {
                    "id": "9809973",
                    "url": "N/A",
                    "categoryID": "N/A",
                    "categoryName": "N/A"
                }
            }
        ],
        "atc": "true",
        "type": "json"
    }
}
```  
**Error_message:** ""Nah nah nah nah nah...this request contains bad characters...""

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * Luyang, Wang (tmacraft@hotmail.com)


