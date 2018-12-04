package com.intel.analytics.zoo.persistent.memory;

public interface BasicMemoryAllocator {
    long allocate(long size);
    void free(long address);
}