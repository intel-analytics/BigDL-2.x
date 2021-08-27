/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class longArray {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected longArray(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(longArray obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_longArray(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public longArray(int nelements) {
    this(swigfaissJNI.new_longArray(nelements), true);
  }

  public int getitem(int index) {
    return swigfaissJNI.longArray_getitem(swigCPtr, this, index);
  }

  public void setitem(int index, int value) {
    swigfaissJNI.longArray_setitem(swigCPtr, this, index, value);
  }

  public SWIGTYPE_p_long cast() {
    long cPtr = swigfaissJNI.longArray_cast(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_long(cPtr, false);
  }

  public static longArray frompointer(SWIGTYPE_p_long t) {
    long cPtr = swigfaissJNI.longArray_frompointer(SWIGTYPE_p_long.getCPtr(t));
    return (cPtr == 0) ? null : new longArray(cPtr, false);
  }

}
