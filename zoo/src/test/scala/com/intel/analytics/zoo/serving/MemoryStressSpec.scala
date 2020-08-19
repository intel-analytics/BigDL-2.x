/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.serving


import com.intel.analytics.zoo.serving.http.{Instances, JsonUtil}
import org.scalatest.{FlatSpec, Matchers}

class MemoryStressSpec extends FlatSpec with Matchers {
  "Stress" should "work" in {
    val inputStr = """{
                     |"instances": [
                     |   {
                     |     "t": [1, 2]
                     |   }
                     |]
                     |}
                     |""".stripMargin
    (0 until 10).foreach(i => {
      val ins = JsonUtil.fromJson(classOf[Instances], inputStr)
      val bytes = ins.toArrow()
      val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
      require(b64 == "/////0gCAAAQAAAAAAAKAA4ABgANAAgACgAAAAAAAwAQAAAAAAEKAAwAAAAIAAQACgAAAAgA" +
        "AAAIAAAAAAAAAAEAAAAEAAAAav7//xQAAAAUAAAA7AEAAAAADQHoAQAAAAAAAAQAAABQAQAA5AAAAHQAAAAEAA" +
        "AAmv7//xQAAAAUAAAATAAAAAAADAFIAAAAAAAAAAEAAAAEAAAAvv7//xQAAAAUAAAAFAAAAAAAAgEYAAAAAAAA" +
        "AAAAAACs/v//AAAAASAAAAAAAAAAAAAAAJT+//8KAAAAaW5kaWNlRGF0YQAABv///xQAAAAUAAAATAAAAAAADA" +
        "FIAAAAAAAAAAEAAAAEAAAAKv///xQAAAAUAAAAFAAAAAAAAgEYAAAAAAAAAAAAAAAY////AAAAASAAAAAAAAAA" +
        "AAAAAAD///8LAAAAaW5kaWNlU2hhcGUAcv///xQAAAAUAAAATAAAAAAADAFIAAAAAAAAAAEAAAAEAAAAlv///x" +
        "QAAAAUAAAAFAAAAAAAAgEYAAAAAAAAAAAAAACE////AAAAASAAAAAAAAAAAAAAAGz///8EAAAAZGF0YQAAAADa" +
        "////FAAAABQAAABoAAAAAAAMAWQAAAAAAAAAAQAAABgAAAAAABIAGAAUABMAEgAMAAAACAAEABIAAAAUAAAAFA" +
        "AAABwAAAAAAAIBIAAAAAAAAAAAAAAACAAMAAgABwAIAAAAAAAAASAAAAAAAAAAAAAAAPD///8FAAAAc2hhcGUA" +
        "AAAEAAQABAAAAAEAAAB0AAAAAAAAAP/////4AQAAFAAAAAAAAAAMABYADgAVABAABAAMAAAASAAAAAAAAAAAAA" +
        "MAEAAAAAADCgAYAAwACAAEAAoAAAAUAAAAKAEAAAEAAAAAAAAAAAAAABEAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAABAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAQAAAAAAAAAAEAAAAAAAAAGAAAAAAAAAAEAAAAAAAAACAAAA" +
        "AAAAAAAQAAAAAAAAAoAAAAAAAAAAwAAAAAAAAAOAAAAAAAAAABAAAAAAAAAEAAAAAAAAAACAAAAAAAAABIAAAA" +
        "AAAAAAAAAAAAAAAASAAAAAAAAAAAAAAAAAAAAEgAAAAAAAAAAAAAAAAAAABIAAAAAAAAAAAAAAAAAAAASAAAAA" +
        "AAAAAAAAAAAAAAAEgAAAAAAAAAAAAAAAAAAABIAAAAAAAAAAAAAAAAAAAASAAAAAAAAAAAAAAAAAAAAAAAAAAJ" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAA" +
        "AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAQAAAAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAAAAAAABAA" +
        "AAAgAAAAAAAAADAAAAAAAAAAEAAAACAAAA/////wAAAAA=", "Your toArrow base64 string is wrong.")
      if (i % 100000 == 0) {
        println(s"$i record to arrow completed. result $b64")
      }
    })
  }

}

