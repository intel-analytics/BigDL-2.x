/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class doubleArray {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected doubleArray(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(doubleArray obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_doubleArray(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public doubleArray(int nelements) {
    this(swigfaissJNI.new_doubleArray(nelements), true);
  }

  public double getitem(int index) {
    return swigfaissJNI.doubleArray_getitem(swigCPtr, this, index);
  }

  public void setitem(int index, double value) {
    swigfaissJNI.doubleArray_setitem(swigCPtr, this, index, value);
  }

  public SWIGTYPE_p_double cast() {
    long cPtr = swigfaissJNI.doubleArray_cast(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_double(cPtr, false);
  }

  public static doubleArray frompointer(SWIGTYPE_p_double t) {
    long cPtr = swigfaissJNI.doubleArray_frompointer(SWIGTYPE_p_double.getCPtr(t));
    return (cPtr == 0) ? null : new doubleArray(cPtr, false);
  }

}
