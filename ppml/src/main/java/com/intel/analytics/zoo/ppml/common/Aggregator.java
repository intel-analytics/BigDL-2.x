package com.intel.analytics.zoo.ppml.common;

import com.intel.analytics.zoo.ppml.generated.FLProto.Table;
import java.util.HashMap;
import java.util.Map;

import static com.intel.analytics.zoo.ppml.common.FLPhase.*;


public abstract class Aggregator {
    public Map<FLPhase, Storage> aggTypeMap;

    public Aggregator() {
        aggTypeMap = new HashMap<>();
        aggTypeMap.put(TRAIN, trainStorage);
        aggTypeMap.put(EVAL, evalStorage);
        aggTypeMap.put(PREDICT, predictStorage);
    }
    public Storage trainStorage = new Storage();
    public Storage evalStorage = new Storage();
    public Storage predictStorage = new Storage();

    protected Integer clientNum;
    public abstract void aggregate();

    public Storage getStorage(FLPhase type) {
        Storage storage = null;
        switch (type) {
            case TRAIN: storage = trainStorage; break;
            case EVAL: storage = evalStorage; break;
            case PREDICT: storage = predictStorage; break;
            default: break;
        }
        return storage;
    }
    public void put(FLPhase type, String clientUUID, int version, Table data)
            throws IllegalArgumentException {
        Storage storage = getStorage(type);
        storage.localData.put(clientUUID, data);

        // Aggregate when buffer is full
        if (storage.localData.size() >= clientNum) {
            aggregate();
        }

    }

}
