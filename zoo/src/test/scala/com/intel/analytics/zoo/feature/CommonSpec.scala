package com.intel.analytics.zoo.feature

import com.intel.analytics.zoo.feature.common.{BigDLAdapter, Preprocessing}
import com.intel.analytics.zoo.feature.image.Resize
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class CommonSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "BigDLAdapter" should "adapt BigDL Transformer" in {
    val newResize = BigDLAdapter(Resize(1, 1))
    assert(newResize.isInstanceOf[Preprocessing[Any, Any]])
  }
}
