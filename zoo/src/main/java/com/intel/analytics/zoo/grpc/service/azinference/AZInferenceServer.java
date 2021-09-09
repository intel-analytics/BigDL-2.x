/*
 * Copyright 2018 Analytics Zoo Authors.
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


package com.intel.analytics.zoo.grpc.service.azinference;

import com.intel.analytics.bigdl.nn.abstractnn.Activity;
import com.intel.analytics.bigdl.tensor.Tensor;
import com.intel.analytics.zoo.grpc.ZooGrpcServer;
import com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceGrpc;
import com.intel.analytics.zoo.grpc.generated.azinference.Content;
import com.intel.analytics.zoo.grpc.generated.azinference.Prediction;
import com.intel.analytics.zoo.grpc.utils.ConfigParser;
import com.intel.analytics.zoo.grpc.utils.EncodeUtils;
import com.intel.analytics.zoo.pipeline.inference.InferenceModel;
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import org.apache.commons.cli.*;

import com.intel.analytics.zoo.grpc.utils.gRPCHelper;

import java.io.IOException;
import java.util.Base64;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * This is an example class to run Analytics Zoo gRPC inference service
 */
public class AZInferenceServer {
    private static final Logger logger = Logger.getLogger(AZInferenceServer.class.getName());

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        Options options = new Options();
        Option portArg = new Option("p", "port", true, "The port to listen.");
        options.addOption(portArg);
        Option configPathArg = new Option("c", "config", true, "The path to config.yaml file");
        options.addOption(configPathArg);
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);

            System.exit(1);
        }
        assert cmd != null;
        int port = Integer.parseInt(cmd.getOptionValue("port", "8083"));
        String configPath = cmd.getOptionValue("config", "config.yaml");

        ZooGrpcServer server = new ZooGrpcServer(new AZInferenceService(configPath));
        server.start();
        server.blockUntilShutdown();
    }

    private static class AZInferenceService extends AZInferenceGrpc.AZInferenceImplBase {
        private final InferenceModel model;
        private int cnt = 0;
        private long time = 0;
        private int pre = 1000;

        AZInferenceService(String configPath) {
            ConfigParser parser = new ConfigParser(configPath);
            gRPCHelper helper = parser.loadConfig();
//            val inputs = (1 to 15).map(i => s"serving_default_input_${i}:0").toArray
            // TODO: to config
            String[] inputs = new String[15];
            for (int i = 1; i <= 15; i ++) {
                inputs[i - 1] = "serving_default_input_" + i + ":0";
            }
            this.model = helper.loadInferenceModel(helper.modelParallelism(), helper.modelPath(),
                    inputs);
//            System.out.println("aa");
        }

        @Override
        public void doPredict(Content request,
                              StreamObserver<Prediction> responseObserver) {
            responseObserver.onNext(predict(request));
            responseObserver.onCompleted();
        }

        private Prediction predict(Content msg) {
            long start = System.nanoTime();
            String encodedStr = msg.getEncodedStr();
//            ------------ JSON input --------------
//            Activity input = JSONSerializer$.MODULE$.deserialize(jsonStr);
//            ------------ Java serialize b64 input -------------
            byte[] bytes1 = Base64.getDecoder().decode(encodedStr);
            Activity input = (Activity) EncodeUtils.bytesToObj(bytes1);
            Activity predictResult = model.doPredict(input);
            PostProcessing post = new PostProcessing((Tensor<Object>) predictResult, "");
            String res = post.tensorToNdArrayString();
//            byte[] bytes = EncodeUtils.objToBytes(predictResult);
//            String b64 = Base64.getEncoder().encodeToString(bytes);
            long end = System.nanoTime();
            if (pre <= 0) {
                time += (end - start);
                cnt += 1;
                if (cnt % 100 == 0) {
                    System.out.println("avg predict time: " + time/cnt);
                }
            } else {
                pre --;
            }
            return Prediction.newBuilder().setPredictStr(res).build();
        }
    }
}
