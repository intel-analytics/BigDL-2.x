### **1. Install**

#### **1.1 Official Release** 

Currently, Analytics Zoo releases are hosted on maven central; here's an example to add the Analytics Zoo dependency to your own project:
```xml
<dependency>
    <groupId>com.intel.analytics.zoo</groupId>
    <artifactId>analytics-zoo-bigdl_0.12.1-[spark_2.1.1|spark_2.2.0|spark_2.3.1|spark_2.4.3|spark_3.0.0]</artifactId>
    <version>${ANALYTICS_ZOO_VERSION}</version>
</dependency>
```
You can find the latest ANALYTICS_ZOO_VERSION [here](https://search.maven.org/search?q=analytics-zoo-bigdl).  

SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.12.1-[spark_2.1.1|spark_2.2.0|spark_2.3.1|spark_2.4.3|spark_3.0.0]" % "${ANALYTICS_ZOO_VERSION}"
```

#### **1.2 Nightly Build**

Currently, Analytics Zoo development version is hosted on [SonaType](https://oss.sonatype.org/content/groups/public/com/intel/analytics/zoo/).

To link your application with the latest Analytics Zoo development version, you should add some dependencies like [official releases](#11-official-release), but set `${ANALYTICS_ZOO_VERSION}` to latest version, and add below repository to your pom.xml.

```xml
<repository>
    <id>sonatype</id>
    <name>sonatype repository</name>
    <url>https://oss.sonatype.org/content/groups/public/</url>
    <releases>
        <enabled>true</enabled>
    </releases>
    <snapshots>
        <enabled>true</enabled>
    </snapshots>
</repository>
```

SBT developers can use
```sbt
resolvers += "ossrh repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
```

#### **1.3 Download Pre-Built Package**

You can download the Analytics Zoo release and nightly build from the [Release Page](../release.md)

### **2. Run**
