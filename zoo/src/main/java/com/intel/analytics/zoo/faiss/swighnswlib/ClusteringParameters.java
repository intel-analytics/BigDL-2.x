/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class ClusteringParameters {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected ClusteringParameters(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(ClusteringParameters obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_ClusteringParameters(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setNiter(int value) {
    swigfaissJNI.ClusteringParameters_niter_set(swigCPtr, this, value);
  }

  public int getNiter() {
    return swigfaissJNI.ClusteringParameters_niter_get(swigCPtr, this);
  }

  public void setNredo(int value) {
    swigfaissJNI.ClusteringParameters_nredo_set(swigCPtr, this, value);
  }

  public int getNredo() {
    return swigfaissJNI.ClusteringParameters_nredo_get(swigCPtr, this);
  }

  public void setVerbose(boolean value) {
    swigfaissJNI.ClusteringParameters_verbose_set(swigCPtr, this, value);
  }

  public boolean getVerbose() {
    return swigfaissJNI.ClusteringParameters_verbose_get(swigCPtr, this);
  }

  public void setSpherical(boolean value) {
    swigfaissJNI.ClusteringParameters_spherical_set(swigCPtr, this, value);
  }

  public boolean getSpherical() {
    return swigfaissJNI.ClusteringParameters_spherical_get(swigCPtr, this);
  }

  public void setInt_centroids(boolean value) {
    swigfaissJNI.ClusteringParameters_int_centroids_set(swigCPtr, this, value);
  }

  public boolean getInt_centroids() {
    return swigfaissJNI.ClusteringParameters_int_centroids_get(swigCPtr, this);
  }

  public void setUpdate_index(boolean value) {
    swigfaissJNI.ClusteringParameters_update_index_set(swigCPtr, this, value);
  }

  public boolean getUpdate_index() {
    return swigfaissJNI.ClusteringParameters_update_index_get(swigCPtr, this);
  }

  public void setFrozen_centroids(boolean value) {
    swigfaissJNI.ClusteringParameters_frozen_centroids_set(swigCPtr, this, value);
  }

  public boolean getFrozen_centroids() {
    return swigfaissJNI.ClusteringParameters_frozen_centroids_get(swigCPtr, this);
  }

  public void setMin_points_per_centroid(int value) {
    swigfaissJNI.ClusteringParameters_min_points_per_centroid_set(swigCPtr, this, value);
  }

  public int getMin_points_per_centroid() {
    return swigfaissJNI.ClusteringParameters_min_points_per_centroid_get(swigCPtr, this);
  }

  public void setMax_points_per_centroid(int value) {
    swigfaissJNI.ClusteringParameters_max_points_per_centroid_set(swigCPtr, this, value);
  }

  public int getMax_points_per_centroid() {
    return swigfaissJNI.ClusteringParameters_max_points_per_centroid_get(swigCPtr, this);
  }

  public void setSeed(int value) {
    swigfaissJNI.ClusteringParameters_seed_set(swigCPtr, this, value);
  }

  public int getSeed() {
    return swigfaissJNI.ClusteringParameters_seed_get(swigCPtr, this);
  }

  public void setDecode_block_size(long value) {
    swigfaissJNI.ClusteringParameters_decode_block_size_set(swigCPtr, this, value);
  }

  public long getDecode_block_size() {
    return swigfaissJNI.ClusteringParameters_decode_block_size_get(swigCPtr, this);
  }

  public ClusteringParameters() {
    this(swigfaissJNI.new_ClusteringParameters(), true);
  }

}
