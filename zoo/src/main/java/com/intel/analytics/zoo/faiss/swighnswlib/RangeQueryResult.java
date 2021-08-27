/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class RangeQueryResult {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected RangeQueryResult(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(RangeQueryResult obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_RangeQueryResult(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setQno(int value) {
    swigfaissJNI.RangeQueryResult_qno_set(swigCPtr, this, value);
  }

  public int getQno() {
    return swigfaissJNI.RangeQueryResult_qno_get(swigCPtr, this);
  }

  public void setNres(long value) {
    swigfaissJNI.RangeQueryResult_nres_set(swigCPtr, this, value);
  }

  public long getNres() {
    return swigfaissJNI.RangeQueryResult_nres_get(swigCPtr, this);
  }

  public void setPres(RangeSearchPartialResult value) {
    swigfaissJNI.RangeQueryResult_pres_set(swigCPtr, this, RangeSearchPartialResult.getCPtr(value), value);
  }

  public RangeSearchPartialResult getPres() {
    long cPtr = swigfaissJNI.RangeQueryResult_pres_get(swigCPtr, this);
    return (cPtr == 0) ? null : new RangeSearchPartialResult(cPtr, false);
  }

  public void add(float dis, int id) {
    swigfaissJNI.RangeQueryResult_add(swigCPtr, this, dis, id);
  }

  public RangeQueryResult() {
    this(swigfaissJNI.new_RangeQueryResult(), true);
  }

}
