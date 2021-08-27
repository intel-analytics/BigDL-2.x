package com.intel.analytics.zoo.faiss.utils;
import com.google.common.base.Preconditions;

import java.io.IOException;

public class JniFaissInitializer {
    private static volatile boolean initialized = false;

    static {
        try {
//            NativeUtils.loadLibraryFromJar("/lib/libmkl_core.so");
//            NativeUtils.loadLibraryFromJar("/lib/libmkl_gnu_thread.so");
//            NativeUtils.loadLibraryFromJar("/lib/libmkl_intel_lp64.so");
            NativeUtils.loadLibraryFromJar("/_swigfaisshnswlib.so");
            initialized = true;
        } catch (IOException e) {
            Preconditions.checkArgument(false);
        }
    }

    public static boolean initialized() {
        return initialized;
    }
}
