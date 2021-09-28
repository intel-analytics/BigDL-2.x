/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class IntersectionCriterion extends AutoTuneCriterion {
  private transient long swigCPtr;

  protected IntersectionCriterion(long cPtr, boolean cMemoryOwn) {
    super(swigfaissJNI.IntersectionCriterion_SWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  protected static long getCPtr(IntersectionCriterion obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_IntersectionCriterion(swigCPtr);
      }
      swigCPtr = 0;
    }
    super.delete();
  }

  public void setR(int value) {
    swigfaissJNI.IntersectionCriterion_R_set(swigCPtr, this, value);
  }

  public int getR() {
    return swigfaissJNI.IntersectionCriterion_R_get(swigCPtr, this);
  }

  public IntersectionCriterion(int nq, int R) {
    this(swigfaissJNI.new_IntersectionCriterion(nq, R), true);
  }

  public double evaluate(SWIGTYPE_p_float D, SWIGTYPE_p_long I) {
    return swigfaissJNI.IntersectionCriterion_evaluate(swigCPtr, this, SWIGTYPE_p_float.getCPtr(D), SWIGTYPE_p_long.getCPtr(I));
  }

}
