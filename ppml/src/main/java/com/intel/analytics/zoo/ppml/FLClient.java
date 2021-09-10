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

package com.intel.analytics.zoo.ppml;

import com.intel.analytics.zoo.grpc.ZooGrpcClient;
import com.intel.analytics.zoo.ppml.generated.FLProto.*;
import com.intel.analytics.zoo.ppml.generated.PSIServiceGrpc;
import com.intel.analytics.zoo.ppml.generated.ParameterServerServiceGrpc;
import com.intel.analytics.zoo.ppml.psi.Utils;
import com.intel.analytics.zoo.ppml.vfl.GBStub;
import com.intel.analytics.zoo.ppml.vfl.NNStub;
import com.intel.analytics.zoo.ppml.vfl.PSIStub;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class FLClient extends ZooGrpcClient {
    private static final Logger logger = LoggerFactory.getLogger(FLClient.class);
    protected String taskID;
    /**
     * All supported FL implementations are listed below
     * VFL includes Private Set Intersection, Neural Network, Gradient Boosting
     */
    public PSIStub psiStub;
    public NNStub nnStub;
    public GBStub gbStub;
    public FLClient() { this(null); }
    public FLClient(String[] args) {
        super(args);
    }

    @Override
    protected void parseConfig() throws IOException {
        FLHelper flHelper = getConfigFromYaml(FLHelper.class, configPath);
        if (flHelper != null) {
            serviceList = flHelper.servicesList;
            target = flHelper.clientTarget;
            taskID = flHelper.taskID;
        }
        super.parseConfig();
    }

    @Override
    public void loadServices() {
        for (String service : serviceList.split(",")) {
            if (service.equals("psi")) {
                psiStub.stub = PSIServiceGrpc.newBlockingStub(channel);
            } else if (service.equals("ps")) {
                // TODO: algorithms stub add here
            } else {
                logger.warn("Type is not supported, skipped. Type: " + service);
            }
        }
    }

    public void shutdown() {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            logger.error("Shutdown Client Error" + e.getMessage());
        }
    }





}
