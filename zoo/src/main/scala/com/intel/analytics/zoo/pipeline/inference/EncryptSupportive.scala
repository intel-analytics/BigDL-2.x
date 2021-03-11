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

import java.io.PrintWriter
import java.util.Base64
import java.security.SecureRandom
import javax.crypto.{Cipher, SecretKeyFactory}
import javax.crypto.spec.{GCMParameterSpec, IvParameterSpec, PBEKeySpec, SecretKeySpec}


trait EncryptSupportive {
  val BLOCK_SIZE = 16

  def encryptWithAESCBC(content: String, secret: String, salt: String,
                        keyLen: Int = 128): String = {
    val iv = new Array[Byte](BLOCK_SIZE)
    val secureRandom: SecureRandom = SecureRandom.getInstance("SHA1PRNG")
    secureRandom.nextBytes(iv)
    val ivParameterSpec = new IvParameterSpec(iv)
    val secretKeyFactory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
    val spec = new PBEKeySpec(secret.toCharArray(), salt.getBytes(), 65536, keyLen)
    val tmp = secretKeyFactory.generateSecret(spec)
    val secretKeySpec = new SecretKeySpec(tmp.getEncoded(), "AES")

    val cipher = Cipher.getInstance("AES/CBC/PKCS5PADDING")
    cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, ivParameterSpec)
    val cipherTextWithoutIV = cipher.doFinal(content.getBytes("UTF-8"))
    Base64.getEncoder().encodeToString(cipher.getIV ++ cipherTextWithoutIV)
  }

  def decryptWithAESCBC(content: String, secret: String, salt: String,
                        keyLen: Int = 128): String = {
    val cipherTextWithIV = Base64.getDecoder.decode(content)
    val iv = cipherTextWithIV.slice(0, BLOCK_SIZE)
    val ivParameterSpec = new IvParameterSpec(iv)
    val secretKeyFactory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
    val spec = new PBEKeySpec(secret.toCharArray(), salt.getBytes(), 65536, keyLen)
    val tmp = secretKeyFactory.generateSecret(spec)
    val secretKeySpec = new SecretKeySpec(tmp.getEncoded(), "AES")

    val cipher = Cipher.getInstance("AES/CBC/PKCS5PADDING")
    cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec)
    val cipherTextWithoutIV = cipherTextWithIV.slice(BLOCK_SIZE, cipherTextWithIV.size)
    new String(cipher.doFinal(cipherTextWithoutIV))
  }

  def encryptWithAESGCM(content: String, secret: String, salt: String,
                        keyLen: Int = 128): String = {
    val iv = new Array[Byte](12)
    val secureRandom: SecureRandom = SecureRandom.getInstance("SHA1PRNG")
    secureRandom.nextBytes(iv)
    val gcmParameterSpec = new GCMParameterSpec(128, iv)
    val secretKeyFactory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
    val spec = new PBEKeySpec(secret.toCharArray(), salt.getBytes(), 65536, keyLen)
    val tmp = secretKeyFactory.generateSecret(spec)
    val secretKeySpec = new SecretKeySpec(tmp.getEncoded(), "AES")

    val cipher = Cipher.getInstance("AES/GCM/NoPadding")
    cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, gcmParameterSpec)
    val cipherTextWithoutIV = cipher.doFinal(content.getBytes("UTF-8"))
    Base64.getEncoder().encodeToString(cipher.getIV ++ cipherTextWithoutIV)
  }

  def decryptWithAESGCM(content: String, secret: String, salt: String,
                        keyLen: Int = 128): String = {
    new String(decryptBytesWithAESGCM(Base64.getDecoder.decode(content), secret,
      salt, keyLen))
  }

  def decryptBytesWithAESGCM(content: Array[Byte], secret: String, salt: String,
                        keyLen: Int = 128): Array[Byte] = {
    val cipherTextWithIV = content
    val iv = cipherTextWithIV.slice(0, 12)
    // 128 means 16 for tag
    val gcmParameterSpec = new GCMParameterSpec(128, iv)
    val secretKeyFactory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
    val spec = new PBEKeySpec(secret.toCharArray(), salt.getBytes(), 65536, keyLen)
    val tmp = secretKeyFactory.generateSecret(spec)
    val secretKeySpec = new SecretKeySpec(tmp.getEncoded(), "AES")

    val cipher = Cipher.getInstance("AES/GCM/NoPadding")
    cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, gcmParameterSpec)
    val cipherTextWithoutIV = cipherTextWithIV.slice(12, cipherTextWithIV.length)
    cipher.doFinal(cipherTextWithoutIV)
  }


  def encryptFileWithAESCBC(filePath: String, secret: String, salt: String, outputFile: String,
                            keyLen: Int = 128, encoding: String = "UTF-8")
  : Unit = {
    val source = scala.io.Source.fromFile(filePath, encoding)
    val content = try source.mkString finally source.close()
    val encrypted = encryptWithAESCBC(content, secret, salt)
    new PrintWriter(outputFile) { write(encrypted); close }
  }

  def decryptFileWithAESCBC(filePath: String, secret: String, salt: String,
                            keyLen: Int = 128, encoding: String = "UTF-8"): String = {
    val source = scala.io.Source.fromFile(filePath, encoding)
    val content = try source.mkString finally source.close()
    decryptWithAESCBC(content, secret, salt, keyLen)
  }

  def decryptFileWithAESCBC(filePath: String, secret: String, salt: String,
                            outputFile: String): Unit = {
    val source = scala.io.Source.fromFile(filePath)
    val content = try source.mkString finally source.close()
    val decrypted = decryptWithAESCBC(content, secret, salt)
    new PrintWriter(outputFile) { write(decrypted); close }
  }

}


