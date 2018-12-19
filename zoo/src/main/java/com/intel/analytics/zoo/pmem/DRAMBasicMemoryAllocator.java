package com.intel.analytics.zoo.pmem;

public class DRAMBasicMemoryAllocator implements BasicMemoryAllocator {
    public static DRAMBasicMemoryAllocator instance = new DRAMBasicMemoryAllocator();
    private DRAMBasicMemoryAllocator() {}
    public long allocate(long size) {
        return org.apache.spark.unsafe.Platform.allocateMemory(size);
    }
    public void free(long address) {
        org.apache.spark.unsafe.Platform.freeMemory(address);
    }
}
