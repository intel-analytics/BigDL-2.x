package com.intel.analytics.zoo.ppml.common;

import com.intel.analytics.zoo.ppml.generated.FLProto;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Storage {
    public int version;
    public FLProto.Table serverData = null;
    public Map<String, FLProto.Table> localData;
    Storage () {
        version = 0;
        localData = new ConcurrentHashMap<>();
    }
}
