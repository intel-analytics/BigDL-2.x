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


package com.intel.analytics.zoo.pipeline.inference

import java.io.File

import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}

class EncryptSpec extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive with EncryptSupportive {

  val plain = "hello world, hello scala, hello encrypt, come on UNITED!!!"
  val secrect = "analytics-zoo"
  val salt = "intel-analytics"

  var tempDir: File = _

  override def beforeAll(): Unit = {
    tempDir = new File(System.getProperty("java.io.tmpdir"))
    println(tempDir)
  }

  override def afterAll(): Unit = {
    tempDir.delete()
  }

  test("plain text should be encrypted") {
    val encrypted = encryptWithAES256(plain, secrect, salt)
    // println(encrypted)
    val decrypted = decryptWithAES256(encrypted, secrect, salt)
    // println(decrypted)
    decrypted should be (plain)
  }

  test("plain file should be encrypted") {
    val file = getClass.getResource("/application.conf")
    val encryptedFile = tempDir.getAbsolutePath + "/" + file.getFile.split("/").last + ".encrpyted"
    println(encryptedFile)
    encryptFileWithAES256(file.getFile, secrect, salt, encryptedFile)
    new File(encryptedFile).exists() should be (true)
  }

}
