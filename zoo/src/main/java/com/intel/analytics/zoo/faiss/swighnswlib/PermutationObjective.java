/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class PermutationObjective {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected PermutationObjective(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(PermutationObjective obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_PermutationObjective(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setN(int value) {
    swigfaissJNI.PermutationObjective_n_set(swigCPtr, this, value);
  }

  public int getN() {
    return swigfaissJNI.PermutationObjective_n_get(swigCPtr, this);
  }

  public double compute_cost(SWIGTYPE_p_int perm) {
    return swigfaissJNI.PermutationObjective_compute_cost(swigCPtr, this, SWIGTYPE_p_int.getCPtr(perm));
  }

  public double cost_update(SWIGTYPE_p_int perm, int iw, int jw) {
    return swigfaissJNI.PermutationObjective_cost_update(swigCPtr, this, SWIGTYPE_p_int.getCPtr(perm), iw, jw);
  }

}
