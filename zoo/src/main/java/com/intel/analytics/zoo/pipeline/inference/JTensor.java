package com.intel.analytics.zoo.pipeline.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JTensor {
  private float[] data;
  private int[] shape;

  public JTensor() {
  }

  public JTensor(List<Float> data, List<Integer> shape) {
    this.data = new float[data.size()];
    for (int i = 0; i < data.size(); i++){
      this.data[i] = data.get(i);
    }
    this.shape = new int[shape.size()];
    for (int i = 0; i < shape.size(); i++){
      this.shape[i] = shape.get(i);
    }
  }

  public JTensor(List<Float> data, Integer[] shape) {
    this.data = new float[data.size()];
    for (int i = 0; i < data.size(); i++){
      this.data[i] = data.get(i);
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(List<Float> data, int[] shape) {
    this.data = new float[data.size()];
    for (int i = 0; i < data.size(); i++){
      this.data[i] = data.get(i);
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(float[] data, List<Integer> shape) {
    this.data = new float[data.length];
    for (int i = 0; i < data.length; i++){
      this.data[i] = data[i];
    }
    this.shape = new int[shape.size()];
    for (int i = 0; i < shape.size(); i++){
      this.shape[i] = shape.get(i);
    }
  }

  public JTensor(float[] data, Integer[] shape) {
    this.data = new float[data.length];
    for (int i = 0; i < data.length; i++){
      this.data[i] = data[i];
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(float[] data, int[] shape) {
    this.data = new float[data.length];
    for (int i = 0; i < data.length; i++){
      this.data[i] = data[i];
    }
    this.shape = new int[shape.length];
    for (int i = 0; i < shape.length; i++){
      this.shape[i] = shape[i];
    }
  }

  public JTensor(float[] data, int[] shape, boolean copy){
    if (copy) {
      this.data = new float[data.length];
      for (int i = 0; i < data.length; i++){
        this.data[i] = data[i];
      }
      this.shape = new int[shape.length];
      for (int i = 0; i < shape.length; i++){
        this.shape[i] = shape[i];
      }
    }
    else {
      this.data = data;
      this.shape = shape;
    }
  }

  public float[] getData() {
    return data;
  }

  public void setData(float[] data) {
    this.data = data;
  }

  public int[] getShape() {
    return shape;
  }

  public void setShape(int[] shape) {
    this.shape = shape;
  }

  @Override
  public String toString() {
    return "JTensor{" +
            "data=" + toString(data) +
            ", shape=" + toString(shape) +
            '}';
  }
  private String toString(float[] d) {
    if (d.length > 100) {
      return "[" + d[0] + "  ...  " + d[d.length - 1] + "]";
    } else {
      String str = "";
      for (int i = 0; i < d.length - 1; i++)
        str = str + d[i] + ' ';
      return "[" + str + d[d.length - 1] + "]";
    }
  }
  private String toString(int[] s) {
    String str = "";
    for (int i = 0; i < s.length - 1; i++)
      str = str + s[i] + ',';
    return "[" + str + s[s.length - 1] + "]";
  }
}