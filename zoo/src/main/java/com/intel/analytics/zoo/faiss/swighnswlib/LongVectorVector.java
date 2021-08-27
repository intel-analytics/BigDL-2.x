/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class LongVectorVector {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected LongVectorVector(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(LongVectorVector obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_LongVectorVector(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public LongVectorVector() {
    this(swigfaissJNI.new_LongVectorVector(), true);
  }

  public void push_back(LongVector arg0) {
    swigfaissJNI.LongVectorVector_push_back(swigCPtr, this, LongVector.getCPtr(arg0), arg0);
  }

  public void clear() {
    swigfaissJNI.LongVectorVector_clear(swigCPtr, this);
  }

  public LongVector data() {
    long cPtr = swigfaissJNI.LongVectorVector_data(swigCPtr, this);
    return (cPtr == 0) ? null : new LongVector(cPtr, false);
  }

  public long size() {
    return swigfaissJNI.LongVectorVector_size(swigCPtr, this);
  }

  public LongVector at(long n) {
    return new LongVector(swigfaissJNI.LongVectorVector_at(swigCPtr, this, n), true);
  }

  public void resize(long n) {
    swigfaissJNI.LongVectorVector_resize(swigCPtr, this, n);
  }

  public void swap(LongVectorVector other) {
    swigfaissJNI.LongVectorVector_swap(swigCPtr, this, LongVectorVector.getCPtr(other), other);
  }

}
