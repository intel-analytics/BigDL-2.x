/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class FileIOWriter extends IOWriter {
  private transient long swigCPtr;

  protected FileIOWriter(long cPtr, boolean cMemoryOwn) {
    super(swigfaissJNI.FileIOWriter_SWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  protected static long getCPtr(FileIOWriter obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_FileIOWriter(swigCPtr);
      }
      swigCPtr = 0;
    }
    super.delete();
  }

  public void setF(SWIGTYPE_p_FILE value) {
    swigfaissJNI.FileIOWriter_f_set(swigCPtr, this, SWIGTYPE_p_FILE.getCPtr(value));
  }

  public SWIGTYPE_p_FILE getF() {
    long cPtr = swigfaissJNI.FileIOWriter_f_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_FILE(cPtr, false);
  }

  public void setNeed_close(boolean value) {
    swigfaissJNI.FileIOWriter_need_close_set(swigCPtr, this, value);
  }

  public boolean getNeed_close() {
    return swigfaissJNI.FileIOWriter_need_close_get(swigCPtr, this);
  }

  public FileIOWriter(SWIGTYPE_p_FILE wf) {
    this(swigfaissJNI.new_FileIOWriter__SWIG_0(SWIGTYPE_p_FILE.getCPtr(wf)), true);
  }

  public FileIOWriter(String fname) {
    this(swigfaissJNI.new_FileIOWriter__SWIG_1(fname), true);
  }

  public int fileno() {
    return swigfaissJNI.FileIOWriter_fileno(swigCPtr, this);
  }

}
