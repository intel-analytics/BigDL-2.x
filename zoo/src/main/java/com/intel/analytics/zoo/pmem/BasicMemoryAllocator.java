package com.intel.analytics.zoo.pmem;

public interface BasicMemoryAllocator {
    long allocate(long size);
    void free(long address);
}