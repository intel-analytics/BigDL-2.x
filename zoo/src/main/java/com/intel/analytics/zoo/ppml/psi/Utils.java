/*
 * Copyright 2021 The Analytics Zoo Authors
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

package com.intel.analytics.zoo.ppml.psi;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;


public class Utils {
    private static final Logger logger = LoggerFactory.getLogger(Utils.class);

    // TODO for XGboost
    public ArrayList<ArrayList<String>> partition(int partitionNum, ArrayList<String> dataset) {
        return new ArrayList<>();
    }

    public static void shuffle(List<String> array,int seed){
        Collections.shuffle(array,new Random(seed));
    }

    /***
     * Gen random HashMap<String, String> for test
     * @param size HashMap size, int
     * @return
     */
    public static HashMap<String, String> genRandomHashSet(int size) {
        HashMap<String, String> data = new HashMap<>();
        Random rand = new Random();
        for (int i = 0; i < size; i++) {
            String name = "User_" + rand.nextInt();
            data.put(name, Integer.toString(i));
        }
        logger.info("IDs are: ");
        for (Map.Entry<String, String> element : data.entrySet()) {
            logger.info(element.getKey() + ",  " + element.getValue());
        }
        return data;
    }

    public static int getTotalSplitNum(List<String> list, int splitSize) {
        return (int)Math.ceil((double)list.size() / splitSize);
    }

    public static List<String> getSplit(List<String> list, int split, int totalSplitNum, int splitSize) {
        if (split < totalSplitNum - 1) {
            return list.subList(split * splitSize, (split + 1) * splitSize);
        } else {
            return list.subList(split * splitSize, list.size());
        }

    }
}
