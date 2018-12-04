package com.intel.analytics.zoo.persistent.memory;

public class PersistentMemoryAllocator implements BasicMemoryAllocator {
    private static volatile PersistentMemoryAllocator instance;

    private PersistentMemoryAllocator() {
        new com.intel.analytics.zoo.persistent.memory.NativeLoader().init();
    }

    public static PersistentMemoryAllocator getInstance() {
        if (instance == null) {
            synchronized (PersistentMemoryAllocator.class) {
                if (instance == null) {
                    instance = new PersistentMemoryAllocator();
                }
            }
        }
        return instance;
    }

    public native void initialize(String path, long size);

    public native long allocate(long size);

    public native void free(long address);

    public native void copy(long destAddress, long srcAddress, long size);
}
