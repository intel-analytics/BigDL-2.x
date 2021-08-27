/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class IndexBinaryHashStats {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected IndexBinaryHashStats(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(IndexBinaryHashStats obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_IndexBinaryHashStats(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setNq(long value) {
    swigfaissJNI.IndexBinaryHashStats_nq_set(swigCPtr, this, value);
  }

  public long getNq() {
    return swigfaissJNI.IndexBinaryHashStats_nq_get(swigCPtr, this);
  }

  public void setN0(long value) {
    swigfaissJNI.IndexBinaryHashStats_n0_set(swigCPtr, this, value);
  }

  public long getN0() {
    return swigfaissJNI.IndexBinaryHashStats_n0_get(swigCPtr, this);
  }

  public void setNlist(long value) {
    swigfaissJNI.IndexBinaryHashStats_nlist_set(swigCPtr, this, value);
  }

  public long getNlist() {
    return swigfaissJNI.IndexBinaryHashStats_nlist_get(swigCPtr, this);
  }

  public void setNdis(long value) {
    swigfaissJNI.IndexBinaryHashStats_ndis_set(swigCPtr, this, value);
  }

  public long getNdis() {
    return swigfaissJNI.IndexBinaryHashStats_ndis_get(swigCPtr, this);
  }

  public IndexBinaryHashStats() {
    this(swigfaissJNI.new_IndexBinaryHashStats(), true);
  }

  public void reset() {
    swigfaissJNI.IndexBinaryHashStats_reset(swigCPtr, this);
  }

}
