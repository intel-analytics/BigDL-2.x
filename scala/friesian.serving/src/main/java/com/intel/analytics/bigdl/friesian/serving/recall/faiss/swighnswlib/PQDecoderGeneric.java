/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class PQDecoderGeneric {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected PQDecoderGeneric(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(PQDecoderGeneric obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_PQDecoderGeneric(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setCode(SWIGTYPE_p_unsigned_char value) {
    swigfaissJNI.PQDecoderGeneric_code_set(swigCPtr, this, SWIGTYPE_p_unsigned_char.getCPtr(value));
  }

  public SWIGTYPE_p_unsigned_char getCode() {
    long cPtr = swigfaissJNI.PQDecoderGeneric_code_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_unsigned_char(cPtr, false);
  }

  public void setOffset(short value) {
    swigfaissJNI.PQDecoderGeneric_offset_set(swigCPtr, this, value);
  }

  public short getOffset() {
    return swigfaissJNI.PQDecoderGeneric_offset_get(swigCPtr, this);
  }

  public int getNbits() {
    return swigfaissJNI.PQDecoderGeneric_nbits_get(swigCPtr, this);
  }

  public long getMask() {
    return swigfaissJNI.PQDecoderGeneric_mask_get(swigCPtr, this);
  }

  public void setReg(short value) {
    swigfaissJNI.PQDecoderGeneric_reg_set(swigCPtr, this, value);
  }

  public short getReg() {
    return swigfaissJNI.PQDecoderGeneric_reg_get(swigCPtr, this);
  }

  public PQDecoderGeneric(SWIGTYPE_p_unsigned_char code, int nbits) {
    this(swigfaissJNI.new_PQDecoderGeneric(SWIGTYPE_p_unsigned_char.getCPtr(code), nbits), true);
  }

  public long decode() {
    return swigfaissJNI.PQDecoderGeneric_decode(swigCPtr, this);
  }

}
