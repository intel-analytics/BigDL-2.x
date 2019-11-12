package com.intel.analytics.zoo.pipeline.api.net;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

/**
 * Developer API for now, intentionally remove public visibility
 */
class PytorchModelWrapper {
  static JTensor[] modelForwardNative(
          long nativeRef, boolean isTraining, float[][] storage, int[] offset, int[][] shape) {
    return (JTensor[]) PytorchModel.modelForwardNative(nativeRef, isTraining,
      storage, offset, shape);
  }

  static JTensor[] modelBackwardNative(
          long nativeRef, float[][] storage, int[] offset, int[][] shape) {
    return (JTensor[]) PytorchModel.modelBackwardNative(nativeRef, storage, offset, shape);
  }

  static JTensor lossForwardNative(
          long nativeRef, float[][] input_storage, int[] input_offset, int[][] input_shape,
          float[][] label_storage, int[] label_offset, int[][] label_shape) {
    return (JTensor) PytorchModel.lossForwardNative(nativeRef, input_storage, input_offset,
     input_shape, label_storage, label_offset, label_shape);
  }

  static JTensor[] lossBackwardNative(long nativeRef) {
    return (JTensor[]) PytorchModel.lossBackwardNative(nativeRef);
  }

}
