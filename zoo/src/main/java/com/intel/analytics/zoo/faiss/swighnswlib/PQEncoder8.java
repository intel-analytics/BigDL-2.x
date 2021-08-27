/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class PQEncoder8 {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected PQEncoder8(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(PQEncoder8 obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_PQEncoder8(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setCode(SWIGTYPE_p_unsigned_char value) {
    swigfaissJNI.PQEncoder8_code_set(swigCPtr, this, SWIGTYPE_p_unsigned_char.getCPtr(value));
  }

  public SWIGTYPE_p_unsigned_char getCode() {
    long cPtr = swigfaissJNI.PQEncoder8_code_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_unsigned_char(cPtr, false);
  }

  public PQEncoder8(SWIGTYPE_p_unsigned_char code, int nbits) {
    this(swigfaissJNI.new_PQEncoder8(SWIGTYPE_p_unsigned_char.getCPtr(code), nbits), true);
  }

  public void encode(long x) {
    swigfaissJNI.PQEncoder8_encode(swigCPtr, this, x);
  }

}
