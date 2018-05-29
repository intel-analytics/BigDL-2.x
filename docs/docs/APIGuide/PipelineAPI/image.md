# Analytics Zoo Image API

Analytics Zoo provides a series of Image APIs for end-to-end image processing pipeline, including image loading, pre-processing, inference/training and some utilities on different formats.

## Load Image
Analytics Zoo provides APIs to read image to different formats:

### Load to Data Frame
Scala:
```scala
package com.intel.analytics.zoo.pipeline.nnframes

object NNImageReader {
  def readImages(path: String, sc: SparkContext, minPartitions: Int = 1): DataFrame
}
```

Read the directory of images from the local or remote source, return DataFrame with a single column "image" of images.

* path: Directory to the input data files, the path can be comma separated paths as the list of inputs. Wildcards path are supported similarly to sc.binaryFiles(path).
* sc: SparkContext to be used.
* minPartitions: Number of the DataFrame partitions, if omitted uses defaultParallelism instead
  
Python:
```python
class zoo.pipeline.nnframes.NNImageReader
    static readImages(path, sc=None, minPartitions=1, bigdl_type="float")
```
### ImageSet
`ImageSet` is a collection of `ImageFeature`. It can be a `DistributedImageSet` for distributed image RDD or
 `LocalImageSet` for local image array.
You can read image data as `ImageSet` from local/distributed image path, or you can directly construct a ImageSet from RDD[ImageFeature] or Array[ImageFeature].

**Scala APIs:**

```scala
object com.intel.analytics.zoo.feature.image.ImageSet
```

```
def array(data: Array[ImageFeature]): LocalImageSet
```
Create LocalImageSet from array of ImeageFeature
  
* data: array of ImageFeature

```
def rdd(data: RDD[ImageFeature]): DistributedImageSet
```
Create DistributedImageSet from rdd of ImageFeature

* data: array of ImageFeature
```
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1): ImageSet
```
Read images as Image Set.
If sc is defined, read image as DistributedImageSet from local file system or HDFS.
If sc is null, Read image as LocalImageSet from local file system

* path: path to read images. If sc is defined, path can be local or HDFS. Wildcard character are supported. If sc is null, path is local directory/image file/image file with wildcard character
* sc: SparkContext
* minPartitions: A suggestion value of the minimal splitting number for input data.
   
Example:
```
// create LocalImageSet from an image folder
val localImageFrame = ImageSet.read("/tmp/image/")

// create DistributedImageFrame from an image folder
val distributedImageFrame2 = ImageSet.read("/tmp/image/", sc, 2)
```

**Python APIs:**

```
class zoo.feature.image.ImageSet
```
```
read(path, sc=None, min_partitions=1, bigdl_type="float")
```
Read images as Image Set.
If sc is defined, read image as DistributedImageSet from local file system or HDFS.
If sc is null, Read image as LocalImageSet from local file system

* path: path to read images. If sc is defined, path can be local or HDFS. Wildcard character are supported. If sc is null, path is local directory/image file/image file with wildcard character
* sc: SparkContext
* minPartitions: A suggestion value of the minimal splitting number for input data.

Python example:
```python
# create LocalImageFrame from an image folder
local_image_frame2 = ImageSet.read("/tmp/image/")

# create DistributedImageFrame from an image folder
distributed_image_frame = ImageSet.read("/tmp/image/", sc, 2)
```
