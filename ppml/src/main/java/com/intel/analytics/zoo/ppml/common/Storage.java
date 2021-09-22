/*
 * Copyright 2021 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
    /**
     *
     * @return The size of data collection of each local node
     */
    public int size() {
        return localData.size();
    }

    /**
     * Put the local data into this storage
     * @param key data key
     * @param value data value
     */
    public void put(String key, FLProto.Table value) {
        localData.put(key, value);
    }
}
