## **Download Analytics Zoo Source**

Analytics Zoo source code is available at [GitHub](https://github.com/intel-analytics/analytics-zoo)

```bash
$ git clone https://github.com/intel-analytics/analytics-zoo.git
```

By default, `git clone` will download the development version of Analytics Zoo, if you want a release version, you can use command `git checkout` to change the version.


## **Setup Build Environment**

The following instructions are aligned with master code.

Maven 3 is needed to build Analytics Zoo, you can download it from the [maven website](https://maven.apache.org/download.cgi).

After installing Maven 3, please set the environment variable MAVEN_OPTS as follows:
```bash
$ export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"
```
When compiling with Java 7, you need to add the option “-XX:MaxPermSize=1G”.


## **Build with script (Recommended)**

It is highly recommended that you build Analytics Zoo using the [make-dist.sh script](https://github.com/intel-analytics/analytics-zoo/blob/master/make-dist.sh). And it will handle the MAVEN_OPTS variable.

Once downloaded, you can build Analytics Zoo with the following commands:
```bash
$ bash make-dist.sh
```
After that, you can find a `dist` folder, which contains all the needed files to run a Analytics Zoo program. The files in `dist` include:

* **dist/lib/analytics-zoo-VERSION-jar-with-dependencies.jar**: This jar package contains all dependencies except Spark classes.
* **dist/lib/analytics-zoo-VERSION-python-api.zip**: This zip package contains all Python files of Analytics Zoo.

## **Build for Spark 2.0 and above**

The instructions above will build Analytics Zoo with Spark 1.5.x or 1.6.x (using Scala 2.10); to build for Spark 2.0 and above (which uses Scala 2.11 by default), pass `-P spark_2.x` to the `make-dist.sh` script:
```bash
$ bash make-dist.sh -P spark_2.x
```

It is highly recommended to use _**Java 8**_ when running with Spark 2.x; otherwise you may observe very poor performance.


## **Build for Scala 2.10 or 2.11**

By default, `make-dist.sh` uses Scala 2.10 for Spark 1.5.x or 1.6.x, and Scala 2.11 for Spark 2.0.x or 2.1.x. To override the default behaviors, you can pass `-P scala_2.10` or `-P scala_2.11` to `make-dist.sh` as appropriate.

---
## **Build with Maven**

To build Analytics Zoo directly using Maven, run the command below:

```bash
$ mvn clean package -DskipTests
```
After that, you can find that jar packages in `PATH_To_Zoo`/target/, where `PATH_To_Zoo` is the path to the directory of the Analytics Zoo.

Note that the instructions above will build Analytics Zoo with Spark 1.5.x or 1.6.x (using Scala 2.10) for Linux. Similarly, you may customize the default behaviors by passing the following parameters to maven:

 - `-P spark_2.x`: build for Spark 2.0 and above (using Scala 2.11). (Again, it is highly recommended to use _**Java 8**_ when running with Spark 2.0; otherwise you may observe very poor performance.)
 * `-P scala_2.10` (or `-P scala_2.11`): build using Scala 2.10 (or Scala 2.11)


---
## **Setup IDE**

We set the scope of spark related library to `provided` in pom.xml. The reason is that we don't want package spark related jars which will make analytics zoo a huge jar, and generally as analytics zoo is invoked by spark-submit, these dependencies will be provided by spark at run-time.

This will cause a problem in IDE. When you run applications, it will throw `NoClassDefFoundError` because the library scope is `provided`.

You can easily change the scopes by the `all-in-one` profile.

* In Intellij, go to View -> Tools Windows -> Maven Projects. Then in the Maven Projects panel, Profiles -> click "all-in-one".
