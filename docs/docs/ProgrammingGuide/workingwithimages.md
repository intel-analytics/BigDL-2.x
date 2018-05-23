# Working with images

Analytics Zoo provides several ways to work with image dataset:

## Read Image
Analytics Zoo provides APIs to read image to different formats:

### Data Frame
Analytics Zoo can process image data as Spark Data Frame.
`NNImageReader` is the primary DataFrame-based image loading interface to read images into DataFrame.

Scala example:
```scala
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.nnframes.NNImageReader

val sc = NNContext.getNNContext("app")
val imageDF1 = NNImageReader.readImages("/tmp", sc)
val imageDF2 = NNImageReader.readImages("/tmp/*.jpg", sc)
val imageDF3 = NNImageReader.readImages("/tmp/a.jpg, /tmp/b.jpg", sc)

```

Python:
```python
from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *

sc = get_nncontext(create_spark_conf().setAppName("app"))
imageDF1 = NNImageReader.readImages("/tmp", sc)
imageDF2 = NNImageReader.readImages("/tmp/*.jpg", sc)
imageDF3 = NNImageReader.readImages("/tmp/a.jpg, /tmp/b.jpg", sc)
```

The output DataFrame contains a sinlge column named "image". The schema of "image" column can be
accessed from `com.intel.analytics.zoo.pipeline.nnframes.DLImageSchema.byteSchema`.
Each record in "image" column represents one image record, in the format of
Row(origin, height, width, num of channels, mode, data), where origin contains the URI for the image file,
and `data` holds the original file bytes for the image file. `mode` represents the OpenCV-compatible
type: CV_8UC3, CV_8UC1 in most cases.
```scala
  val byteSchema = StructType(
    StructField("origin", StringType, true) ::
      StructField("height", IntegerType, false) ::
      StructField("width", IntegerType, false) ::
      StructField("nChannels", IntegerType, false) ::
      // OpenCV-compatible type: CV_8UC3, CV_32FC3 in most cases
      StructField("mode", IntegerType, false) ::
      // Bytes in OpenCV-compatible order: row-wise BGR in most cases
      StructField("data", BinaryType, false) :: Nil)
```

After loading the image, user can compose the preprocess steps with the `Preprocessing` defined
in `com.intel.analytics.zoo.feature.image`.

### ImageSet
Analytics Zoo can read image data as ImageSet. There are two kinds of ImagSet, Distri