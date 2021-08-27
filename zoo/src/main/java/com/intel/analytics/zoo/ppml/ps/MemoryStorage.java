/*
 * Copyright 2021 The Analytic Zoo Authors
 *
 * Licensed under the Apache License,  Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,  software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.analytics.zoo.ppml.ps;

import com.intel.analytics.zoo.ppml.generated.FLProto.Table;
import com.intel.analytics.zoo.ppml.generated.FLProto.TableMetaData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class MemoryStorage extends Storage {
    protected Map<Integer, Table> memMap;
    protected static final Logger logger = LoggerFactory.getLogger(MemoryStorage.class);
    // key is the version, maybe List is better
    protected int currentVersion;

    public MemoryStorage() {
        memMap = new HashMap<>();
    }

    @Override
    public Integer retrieveCurrentVersion() {
        return currentVersion;
    }

    @Override
    public void save(Integer version, Table data) {
        currentVersion = version;
        synchronized (memMap) {
            memMap.put(version, data);
            memMap.notifyAll();
        }
        logger.info("The memory store the following data");
        TableMetaData metaData = data.getMetaData();
        logger.info("Saved data name " + metaData.getName());
        logger.info("Saved data version " + metaData.getVersion());
        logger.debug("Saved data " + data.getTableMap());
    }

    @Override
    public Table retrieve(Integer version) {
        Table data;
        synchronized (memMap) {
            data = memMap.get(version);
            if (null == data) {
                try {
                    memMap.wait();
                    data = memMap.get(version);
                } catch (InterruptedException ie) {
                    throw new RuntimeException(ie);
                }
            }
        }
        return data;
    }
}
