/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class EnumeratedVectors {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected EnumeratedVectors(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(EnumeratedVectors obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_EnumeratedVectors(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setNv(long value) {
    swigfaissJNI.EnumeratedVectors_nv_set(swigCPtr, this, value);
  }

  public long getNv() {
    return swigfaissJNI.EnumeratedVectors_nv_get(swigCPtr, this);
  }

  public void setDim(int value) {
    swigfaissJNI.EnumeratedVectors_dim_set(swigCPtr, this, value);
  }

  public int getDim() {
    return swigfaissJNI.EnumeratedVectors_dim_get(swigCPtr, this);
  }

  public long encode(SWIGTYPE_p_float x) {
    return swigfaissJNI.EnumeratedVectors_encode(swigCPtr, this, SWIGTYPE_p_float.getCPtr(x));
  }

  public void decode(long code, SWIGTYPE_p_float c) {
    swigfaissJNI.EnumeratedVectors_decode(swigCPtr, this, code, SWIGTYPE_p_float.getCPtr(c));
  }

  public void encode_multi(long nc, SWIGTYPE_p_float c, SWIGTYPE_p_unsigned_long codes) {
    swigfaissJNI.EnumeratedVectors_encode_multi(swigCPtr, this, nc, SWIGTYPE_p_float.getCPtr(c), SWIGTYPE_p_unsigned_long.getCPtr(codes));
  }

  public void decode_multi(long nc, SWIGTYPE_p_unsigned_long codes, SWIGTYPE_p_float c) {
    swigfaissJNI.EnumeratedVectors_decode_multi(swigCPtr, this, nc, SWIGTYPE_p_unsigned_long.getCPtr(codes), SWIGTYPE_p_float.getCPtr(c));
  }

  public void find_nn(long n, SWIGTYPE_p_unsigned_long codes, long nq, SWIGTYPE_p_float xq, SWIGTYPE_p_long idx, SWIGTYPE_p_float dis) {
    swigfaissJNI.EnumeratedVectors_find_nn(swigCPtr, this, n, SWIGTYPE_p_unsigned_long.getCPtr(codes), nq, SWIGTYPE_p_float.getCPtr(xq), SWIGTYPE_p_long.getCPtr(idx), SWIGTYPE_p_float.getCPtr(dis));
  }

}
