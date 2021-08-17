## Migration Guidance

This guidance is used to provide guidance to BigDL and Analytics zoo users to migrate their existing BigDL/Analytics Zoo applications to use BigDL2.0

* **For BigDL users**

   ***scala application***

   Change ```import com.intel.analytics.bigdl.XYZ``` to ```import com.intel.analytics.bigdl.dllib.XYZ```

    except the following:

   ```com.intel.analytics.bigdl.dataset.XYZ``` to ```com.intel.analytics.bigdl.dllib.feature.dataset.XYZ```

   ```com.intel.analytics.bigdl.transform.XYZ``` to ```com.intel.analytics.bigdl.dllib.feature.transform.XYZ```
   
   ```com.intel.analytics.bigdl.nn.keras.XYZ``` is deprecated and will be removed. Pleaase use zoo keras api instead


   And below classes remain the same:

   ```com.intel.analytics.bigdl.utils.Engine```

   ```com.intel.analytics.bigdl.utils.LoggerFilter```

   ```com.intel.analytics.bigdl.utils.RandomGenerator```

   ```com.intel.analytics.bigdl.utils.Shape```

   If you are a maven user and add BigDL as dependency to your own project. Please change the dependency as :
   ```
   <dependency>
       <groupId>com.intel.analytics.bigdl</groupId>
       <artifactId>bigdl-dllib-SPARAK_2.4</artifactId>
       <version>${BIGDL_VERSION}</version>
   </dependency>
   ```

   If you are a sbt user, please change libraryDependencies to:
   ```
   libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-dllib-SPARK_2.4" % "${BIGDL_VERSION}"
   ```

   ***python application***


* **For Analytics Zoo users**
