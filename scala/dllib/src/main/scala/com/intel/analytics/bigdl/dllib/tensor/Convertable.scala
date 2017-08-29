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

package com.intel.analytics.bigdl.tensor

/**
 * This package is used to provide concrete implementations of the conversions
 * between numeric primitives. The idea here is that the Numeric trait can
 * extend these traits to inherit the conversions.
 *
 * We can also use these implementations to provide a way to convert from
 * A -> B, where both A and B are generic Numeric types. Without a separate
 * trait, we'd have circular type definitions when compiling Numeric.
 */

import scala.language.implicitConversions
import scala.{specialized => spec}

/**
 * Conversions to type.
 *
 * An object implementing ConvertableTo[A] provides methods to go
 * from number types to A.
 */
trait ConvertableTo[@spec A] {
  implicit def fromFloat(a: Float): A

  implicit def fromDouble(a: Double): A

  implicit def fromInt(a: Int): A
}

trait ConvertableToFloat extends ConvertableTo[Float] {
  implicit def fromFloat(a: Float): Float = a

  implicit def fromDouble(a: Double): Float = a.toFloat

  implicit def fromInt(a: Int): Float = a.toFloat
}

trait ConvertableToDouble extends ConvertableTo[Double] {
  implicit def fromFloat(a: Float): Double = a.toDouble

  implicit def fromDouble(a: Double): Double = a

  implicit def fromInt(a: Int): Double = a.toDouble
}

trait ConvertableToInt extends ConvertableTo[Int] {
  implicit def fromFloat(a: Float): Int = a.toInt

  implicit def fromDouble(a: Double): Int = a.toInt

  implicit def fromInt(a: Int): Int = a
}

object ConvertableTo {

  implicit object ConvertableToFloat extends ConvertableToFloat

  implicit object ConvertableToDouble extends ConvertableToDouble

  implicit object ConvertableToInt extends ConvertableToInt

}


/**
 * Conversions from type.
 *
 * An object implementing ConvertableFrom[A] provides methods to go
 * from A to number types (and String).
 */
trait ConvertableFrom[@spec A] {
  implicit def toFloat(a: A): Float

  implicit def toDouble(a: A): Double

  implicit def toShort(a: A): Short

  implicit def toLong(a: A): Long

  implicit def toInt(a: A): Int

  implicit def toString(a: A): String

  implicit def toChar(a: A): Char

  implicit def toBoolean(a: A): Boolean
}

trait ConvertableFromFloat extends ConvertableFrom[Float] {
  implicit def toFloat(a: Float): Float = a

  implicit def toDouble(a: Float): Double = a.toDouble

  implicit def toInt(a: Float): Int = a.toInt

  implicit def toShort(a: Float): Short = a.toShort

  implicit def toLong(a: Float): Long = a.toLong

  implicit def toString(a: Float): String = a.toString

  implicit def toChar(a: Float): Char = a.toChar

  implicit def toBoolean(a: Float): Boolean =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")
}

trait ConvertableFromDouble extends ConvertableFrom[Double] {
  implicit def toFloat(a: Double): Float = a.toFloat

  implicit def toDouble(a: Double): Double = a

  implicit def toInt(a: Double): Int = a.toInt

  implicit def toShort(a: Double): Short = a.toShort

  implicit def toLong(a: Double): Long = a.toLong

  implicit def toString(a: Double): String = a.toString

  implicit def toChar(a: Double): Char = a.toChar

  implicit def toBoolean(a: Double): Boolean =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")
}

trait ConvertableFromInt extends ConvertableFrom[Int] {
  implicit def toFloat(a: Int): Float = a.toFloat

  implicit def toDouble(a: Int): Double = a.toDouble

  implicit def toInt(a: Int): Int = a

  implicit def toShort(a: Int): Short = a.toShort

  implicit def toLong(a: Int): Long = a.toLong

  implicit def toString(a: Int): String = a.toString

  implicit def toChar(a: Int): Char = a.toChar

  implicit def toBoolean(a: Int): Boolean =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")
}

trait ConvertableFromShort extends ConvertableFrom[Short] {
  implicit def toFloat(a: Short): Float = a.toFloat

  implicit def toDouble(a: Short): Double = a.toDouble

  implicit def toInt(a: Short): Int = a

  implicit def toShort(a: Short): Short = a.toShort

  implicit def toLong(a: Short): Long = a.toLong

  implicit def toString(a: Short): String = a.toString

  implicit def toChar(a: Short): Char = a.toChar

  implicit def toBoolean(a: Short): Boolean =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")
}

trait ConvertableFromLong extends ConvertableFrom[Long] {
  implicit def toFloat(a: Long): Float = a.toFloat

  implicit def toDouble(a: Long): Double = a.toDouble

  implicit def toInt(a: Long): Int = a

  implicit def toShort(a: Long): Short = a.toShort

  implicit def toLong(a: Long): Long = a.toLong

  implicit def toString(a: Long): String = a.toString

  implicit def toChar(a: Long): Char = a.toChar

  implicit def toBoolean(a: Long): Boolean =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")
}

trait ConvertableFromBoolean {
  implicit def toFloat(a: Boolean): Float =
    throw new UnsupportedOperationException("Boolean cannot be cast to Float")

  implicit def toDouble(a: Boolean): Double =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")

  implicit def toInt(a: Boolean): Int =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")

  implicit def toShort(a: Boolean): Short =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")

  implicit def toLong(a: Boolean): Long =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean type")

  implicit def toString(a: Boolean): String = a.toString

  implicit def toChar(a: Boolean): Char =
    throw new UnsupportedOperationException("Float cannot be cast to Boolean")

  implicit def toBoolean(a: Boolean): Boolean = a
}

trait ConvertableFromString {
  implicit def toFloat(a: String): Float =
    throw new UnsupportedOperationException("Float cannot be cast to String")

  implicit def toDouble(a: String): Double =
    throw new UnsupportedOperationException("Double cannot be cast to String")

  implicit def toInt(a: String): Int =
    throw new UnsupportedOperationException("Int cannot be cast to String")

  implicit def toShort(a: String): Short =
    throw new UnsupportedOperationException("Short cannot be cast to String")

  implicit def toLong(a: String): Long =
    throw new UnsupportedOperationException("Long cannot be cast to String")

  implicit def toChar(a: String): Char =
    throw new UnsupportedOperationException("Char cannot be cast to char type")

  implicit def toBoolean(a: String): Boolean =
    throw new UnsupportedOperationException("Boolean cannot be cast to String")

  implicit def toString(a: String): String = a
}

trait ConvertableFromChar {
  implicit def toFloat(a: Char): Float = a.toFloat

  implicit def toDouble(a: Char): Double = a.toDouble

  implicit def toInt(a: Char): Int = a.toInt

  implicit def toShort(a: Char): Short = a.toShort

  implicit def toLong(a: Char): Long = a.toLong

  implicit def toBoolean(a: Char): Boolean =
    throw new UnsupportedOperationException("Char cannot be cast to boolean type")

  implicit def toString(a: Char): String = a.toString

  implicit def toChar(a: Char): Char = a
}

object ConvertableFrom {

  implicit object ConvertableFromFloat extends ConvertableFromFloat

  implicit object ConvertableFromDouble extends ConvertableFromDouble

  implicit object ConvertableFromInt extends ConvertableFromInt

  implicit object ConvertableFromChar extends ConvertableFromChar

  implicit object ConvertableFromShort extends ConvertableFromShort

  implicit object ConvertableFromLong extends ConvertableFromLong

  implicit object ConvertableFromString extends ConvertableFromString
}

