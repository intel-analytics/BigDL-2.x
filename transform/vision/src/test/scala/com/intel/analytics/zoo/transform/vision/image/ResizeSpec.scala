/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.transform.vision.image

import java.io.ByteArrayInputStream
import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.zoo.transform.vision.image.augmentation.Resize
import org.scalatest.FlatSpec

class ResizeSpec extends FlatSpec {
  "resize performance compare" should "work" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = Files.readAllBytes(Paths.get(resource.getFile))

    val r = 500
    val byteImage2 = Array(ImageFeature(img)).toIterator.flatMap(x => {
      (1 to 10).toIterator.map(i => x(ImageFeature.bytes).asInstanceOf[Array[Byte]])
    })

    val reize = new RoiImageResizer(r)
    val start2 = System.nanoTime()
    val outw = reize(byteImage2).foreach(x => {

    }
    )

    println("java takes " + (System.nanoTime() - start2) / 1e9)

    val byteImage = Array(ImageFeature(img)).toIterator.flatMap(x => {
      (1 to 10).toIterator.map(i => x)
    })
    val imgAug = BytesToMat() ->
      Resize(r, r, -1) ->
      MatToFloats(validHeight = 300, validWidth = 300)
    val start = System.nanoTime()
    val out = imgAug(byteImage)
    out.foreach(img => {
    })
    println("opencv takes " + (System.nanoTime() - start) / 1e9)



  }
}


class RoiImageResizer(scale: Int)
  extends Transformer[Array[Byte], Array[Float]] {


  override def apply(prev: Iterator[Array[Byte]]): Iterator[Array[Float]] = {
    prev.map(roiByteImage => {
      transform(roiByteImage)
    })
  }

  val floats = new Array[Float](scale * scale * 3)

  def transform(roiByteImage: Array[Byte]): Array[Float] = {
    // convert byte array back to BufferedImage
    val img = ImageIO.read(new ByteArrayInputStream(roiByteImage,
      0, roiByteImage.length))
    val (height, width) = (scale, scale)
    val rawData = BGRImage.resizeImage(img, width, height)
    var i = 0
    while (i < width * height * 3) {
      floats(i) = rawData(i + 8) & 0xff
      i += 1
    }
    floats
  }

}
