package com.intel.analytics.zoo.serving.preprocessing

object DataType extends Enumeration{
  val IMAGE = value("IMAGE", 0)
  val TENSOR = value("TENSOR", 1)
  val SPARSETENSOR = value("SPARSETENSOR", 2)

  class DataTypeEnumVal(name: String, val value: Int) extends Val(nextId, name)

  protected final def value(name: String, value: Int): DataTypeEnumVal =
    new DataTypeEnumVal(name, value)
}
