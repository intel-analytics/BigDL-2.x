package com.intel.analytics.zoo.friesian.service.azinference;

import com.intel.analytics.bigdl.nn.abstractnn.Activity;
import com.intel.analytics.bigdl.tensor.Tensor;
import com.intel.analytics.zoo.friesian.generated.azinference.AZInferenceGrpc;
import com.intel.analytics.zoo.friesian.generated.azinference.Content;
import com.intel.analytics.zoo.friesian.generated.azinference.Prediction;
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing;
import com.intel.analytics.zoo.friesian.utils.EncodeUtils;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

import java.io.*;
import java.util.concurrent.TimeUnit;
import org.apache.log4j.Logger;
import java.util.Base64;
import com.intel.analytics.zoo.pipeline.inference.InferenceModel;
import org.apache.commons.cli.*;
import com.intel.analytics.zoo.friesian.utils.ConfigParser;
import com.intel.analytics.zoo.friesian.utils.gRPCHelper;


public class AZInferenceServer {
    private static final Logger logger = Logger.getLogger(AZInferenceServer.class.getName());

    private final int port;
    private final Server server;

    /** Create an AZInference server listening on {@code port} using {@code configPath} config. */
    public AZInferenceServer(int port, String configPath) {
        this(ServerBuilder.forPort(port), port, configPath);
    }

    /** Create an AZInference server using serverBuilder as a base. */
    public AZInferenceServer(ServerBuilder<?> serverBuilder, int port, String configPath) {
        this.port = port;
        server = serverBuilder.addService(new AZInferenceService(configPath))
                .build();
    }

    /** Start serving requests. */
    public void start() throws IOException {
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                try {
                    AZInferenceServer.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
                System.err.println("*** server shut down");
            }
        });
    }

    /** Stop serving requests and shutdown resources. */
    public void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }

    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

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
        AZInferenceServer server = new AZInferenceServer(port, configPath);
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
            this.model = helper.loadInferenceModel(helper.modelParallelism(), helper.modelPath(),
                    helper.savedModelInputsArr());
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
            byte[] bytes1 = Base64.getDecoder().decode(encodedStr);
            Activity input = (Activity) EncodeUtils.bytesToObj(bytes1);
            Activity predictResult = model.doPredict(input);
            PostProcessing post = new PostProcessing((Tensor<Object>) predictResult, "");
            String res = post.tensorToNdArrayString();
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
