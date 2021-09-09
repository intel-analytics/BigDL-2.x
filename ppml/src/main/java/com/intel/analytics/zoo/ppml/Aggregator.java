package com.intel.analytics.zoo.ppml;

import com.intel.analytics.zoo.ppml.generated.FLProto.Table;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
public class Storage {
    int version = 0;
    Table serverData = null;
    Map<String, Table> localData = new ConcurrentHashMap<>();
}
public abstract class Aggregator {
    static final int TRAIN = 0;
    static final int EVAL = 1;
    static final int PREDICT = 2;
    Map<Integer, Storage> aggTypeMap;

    Aggregator() {
        aggTypeMap = new HashMap<>();
        aggTypeMap.put(TRAIN, trainStorage);
        aggTypeMap.put(EVAL, evalStorage);
        aggTypeMap.put(PREDICT, predictStorage);
    }
    protected Storage trainStorage = new Storage();
    protected Storage evalStorage = new Storage();
    protected Storage predictStorage = new Storage();

    protected Integer clientNum;
    public abstract void aggregate();

    protected Storage getStorage(int type) {
        Storage storage = null;
        switch (type) {
            case TRAIN: storage = trainStorage; break;
            case EVAL: storage = evalStorage; break;
            case PREDICT: storage = predictStorage; break;
            default: break;
        }
        return storage;
    }
    public void put(int type, String clientUUID, int version, Table data)
            throws IllegalArgumentException {
        Storage storage = getStorage(type);
        storage.localData.put(clientUUID, data);

        // Aggregate when buffer is full
        if (storage.localData.size() >= clientNum) {
            aggregate();
        }

    }

}
