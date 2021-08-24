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

package com.intel.analytics.zoo.ppml.psi.test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static com.intel.analytics.zoo.ppml.psi.SecurityUtils.int2Bytes;
import static com.intel.analytics.zoo.ppml.psi.SecurityUtils.toHexString;

public class ExampleUtils {
    private static final Logger logger = LoggerFactory.getLogger(ExampleUtils.class);
    public static HashMap<String, String> getRandomHashSetOfStringForFiveFixed(int size) {
        HashMap<String, String> data = new HashMap<>();
        Random rand = new Random();
        // put several constant for test
        String nameTest = "User_11111111111111111111111111111";//randomBytes;
        data.put(nameTest, Integer.toString(0));
        nameTest = "User_111111111111111111111111122222";//randomBytes;
        data.put(nameTest, Integer.toString(1));
        nameTest = "User_11111111111111111111111133333";//randomBytes;
        data.put(nameTest, Integer.toString(2));
        nameTest = "User_11111111111111111111111144444";//randomBytes;
        data.put(nameTest, Integer.toString(3));
        nameTest = "User_11111111111111111111111155555";//randomBytes;
        data.put(nameTest, Integer.toString(4));
        for (int i = 5; i < size; i++) {
            //String randomBytes = new String(getSecurityRandomBytes());
            String name = toHexString(int2Bytes(i));//randomBytes;
            data.put(name, Integer.toString(i));
        }
        logger.info("IDs are: ");
        for (Map.Entry<String, String> element : data.entrySet()) {
            logger.info(element.getKey() + ",  " + element.getValue());
        }
        return data;
    }
}
