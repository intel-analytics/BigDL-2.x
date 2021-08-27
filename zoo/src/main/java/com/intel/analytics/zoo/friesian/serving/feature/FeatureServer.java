package com.intel.analytics.zoo.friesian.service.feature;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.zoo.friesian.generated.feature.FeatureServiceGrpc;
import com.intel.analytics.zoo.friesian.generated.feature.Features;
import com.intel.analytics.zoo.friesian.generated.feature.IDs;
import com.intel.analytics.zoo.friesian.generated.feature.ServerMessage;
import com.intel.analytics.zoo.friesian.utils.feature.RedisUtils;
import com.intel.analytics.zoo.serving.http.JacksonJsonSerializer;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;

import java.io.*;
import java.util.*;
import java.util.concurrent.TimeUnit;

import java.util.stream.Collectors;

import com.intel.analytics.zoo.pipeline.inference.InferenceModel;
import org.apache.commons.cli.*;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;
import redis.clients.jedis.*;
import com.intel.analytics.zoo.friesian.utils.feature.FeatureUtils;
import com.intel.analytics.zoo.friesian.utils.TimerMetrics;
import com.intel.analytics.zoo.friesian.utils.TimerMetrics$;
import com.intel.analytics.zoo.friesian.utils.ConfigParser;

enum ServiceType {
    KV, INFERENCE
}

enum SearchType {
    ITEM, USER
}

public class FeatureServer {
    private static final Logger logger = Logger.getLogger(FeatureServer.class.getName());

    private final int port;
    private final Server server;


    /** Create a Feature server listening on {@code port} using {@code configPath} config. */
    public FeatureServer(int port, String configPath) throws Exception {
        this(ServerBuilder.forPort(port), port, configPath);
    }

    /** Create a Feature server using serverBuilder as a base. */
    public FeatureServer(ServerBuilder<?> serverBuilder, int port, String configPath) throws Exception {
        this.port = port;
        Logger.getLogger("org.apache").setLevel(Level.ERROR);
        server = serverBuilder.addService(new FeatureService(configPath))
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
                    FeatureServer.this.stop();
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
        int port = Integer.parseInt(cmd.getOptionValue("port", "8082"));
        String configPath = cmd.getOptionValue("config", "config.yaml");
        FeatureServer server = new FeatureServer(port, configPath);
        server.start();
        server.blockUntilShutdown();
    }

    private static class FeatureService extends FeatureServiceGrpc.FeatureServiceImplBase {
        private InferenceModel userModel;
        private InferenceModel itemModel;
        private RedisUtils redis;
        private final boolean redisCluster;
        private Set<ServiceType> serviceType;
        private MetricRegistry metrics = new MetricRegistry();
        Timer overallTimer = metrics.timer("feature.overall");
        Timer userPredictTimer = metrics.timer("feature.user.predict");
        Timer itemPredictTimer = metrics.timer("feature.item.predict");
        Timer redisTimer = metrics.timer("feature.redis");

        FeatureService(String configPath) throws Exception {
            ConfigParser parser = new ConfigParser(configPath);
            FeatureUtils.helper_$eq(parser.loadConfig());
            serviceType = new HashSet<>();
            parseServiceType();
            if (serviceType.contains(ServiceType.KV)) {
                redis = RedisUtils.getInstance();
                if (FeatureUtils.helper().getLoadInitialData()) {
                    // Load features in files
                    SparkSession spark = SparkSession.builder().getOrCreate();
                    FeatureUtils.loadUserItemFeaturesRDD(spark);
                }
            }
            redisCluster = (redis.getCluster() != null) ? true : false;

            if (serviceType.contains(ServiceType.INFERENCE)) {
                if (FeatureUtils.helper().getUserModelPath() != null) {
                    userModel = FeatureUtils.helper()
                            .loadInferenceModel(FeatureUtils.helper().getModelParallelism(),
                                    FeatureUtils.helper().getUserModelPath(), null);
                }
                if (FeatureUtils.helper().getItemModelPath() != null){
                    itemModel = FeatureUtils.helper()
                            .loadInferenceModel(FeatureUtils.helper().getModelParallelism(),
                                    FeatureUtils.helper().getItemModelPath(), null);
                }
                if (userModel == null && itemModel == null) {
                    throw new Exception("Either userModelPath or itemModelPath should be provided.");
                }
            }

        }

        void parseServiceType() {
            Map<String, ServiceType> typeMap = new HashMap<String, ServiceType>() {{
                put("kv", ServiceType.KV);
                put("inference", ServiceType.INFERENCE);
            }};
            String[] typeArray = FeatureUtils.helper().serviceType().split("\\s*,\\s*");
            for (String typeStr : typeArray) {
                serviceType.add(typeMap.get(typeStr));
            }
        }
        @Override
        public void getUserFeatures(IDs request,
                                    StreamObserver<Features> responseObserver) {
            Features result;
            try {
                result = getFeatures(request, SearchType.USER);
            } catch (Exception e) {
                e.printStackTrace();
                logger.warn(e.getMessage());
                responseObserver.onError(Status.INTERNAL.withDescription(e.getMessage())
                        .asRuntimeException());
                return;
            }
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }

        @Override
        public void getItemFeatures(IDs request,
                                    StreamObserver<Features> responseObserver) {
            Features result;
            try {
                result = getFeatures(request, SearchType.ITEM);
            } catch (Exception e) {
                e.printStackTrace();
                logger.warn(e.getMessage());
                responseObserver.onError(Status.INTERNAL.withDescription(e.getMessage())
                        .asRuntimeException());
                return;
            }
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<com.intel.analytics.zoo.friesian.generated.feature.ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
        }

        private Features getFeatures(IDs msg, SearchType searchType) throws Exception {
            Timer.Context overallContext = overallTimer.time();
            Features result;
            if (serviceType.contains(ServiceType.KV) && serviceType.contains(ServiceType.INFERENCE)) {
                result = getFeaturesFromRedisAndInference(msg, searchType);
            }
            else if (serviceType.contains(ServiceType.KV)) {
                result = getFeaturesFromRedis(msg, searchType);
            } else if (serviceType.contains(ServiceType.INFERENCE)){
                result = getFeaturesFromInferenceModel(msg, searchType);
            } else {
                throw new Exception("ServiceType is not supported, only 'kv', 'inference' and " +
                        "'kv, inference' are supported");
            }

            overallContext.stop();

            return result;
        }

        private Features getFeaturesFromRedisAndInference(IDs msg, SearchType searchType) throws Exception {
            Features.Builder featureBuilder = Features.newBuilder();
            Timer.Context predictContext;
            String typeStr = "";
            InferenceModel model;
            String[] featureCols;
            String idCol;
            Features userFeatures = getFeaturesFromRedis(msg, searchType);
            if (searchType == SearchType.USER) {
                predictContext = userPredictTimer.time();
                model = this.userModel;
                typeStr = "user";
                featureCols = FeatureUtils.helper().userFeatureColArr();
                idCol = FeatureUtils.helper().userIDColumn();
            } else {
                predictContext = itemPredictTimer.time();
                model = this.itemModel;
                typeStr = "item";
                featureCols = FeatureUtils.helper().itemFeatureColArr();
                idCol = FeatureUtils.helper().itemIDColumn();
            }
            if (model == null) {
                throw new Exception(typeStr + "ModelPath should be provided in the config.yaml " +
                        "file");
            }
            List<String> result = FeatureUtils.predictFeatures(userFeatures, model,
                    featureCols, idCol);
            for (String feature: result) {
                featureBuilder.addB64Feature(feature);
            }
            predictContext.close();
            return featureBuilder.build();
        }

        private Features getFeaturesFromRedis(IDs msg, SearchType searchType) {
            String keyPrefix = searchType == SearchType.USER ? "userid": "itemid";
            List<Integer> ids = msg.getIDList();
            Jedis jedis = redisCluster ? redis.getRedisClient() : null;

            Features.Builder featureBuilder = Features.newBuilder();
            for (int id : ids) {
                Timer.Context redisContext = redisTimer.time();
                String value;
                if (!redisCluster) {
                    value = jedis.hget(keyPrefix + ":" + id, "value");
                } else {
                    value = redis.getCluster().hget(keyPrefix + ":" + id, "value");
                }

//                Set<String> value = jedis.keys(keyPrefix + ":" + id);km/........................,

                redisContext.stop();
                if (value == null) {
                    value = "";
                }
                featureBuilder.addB64Feature(value);
            }
            jedis.close();
            return featureBuilder.build();
        }

        private Features getFeaturesFromInferenceModel(IDs msg, SearchType searchType) throws Exception {
            Features.Builder featureBuilder = Features.newBuilder();
            Timer.Context predictContext;
            String typeStr = "";
            InferenceModel model;
            if (searchType == SearchType.USER) {
                predictContext = userPredictTimer.time();
                model = this.userModel;
                typeStr = "user";
            } else {
                predictContext = itemPredictTimer.time();
                model = this.itemModel;
                typeStr = "item";
            }
            if (model == null) {
                throw new Exception(typeStr + "ModelPath should be provided in the config.yaml " +
                        "file");
            }
            List<String> result = FeatureUtils.doPredict(msg, model);
            for (String feature: result) {
                featureBuilder.addB64Feature(feature);
            }
            predictContext.close();
            return featureBuilder.build();
        }

        private ServerMessage getMetrics() {
            JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
            Set<String> keys = metrics.getTimers().keySet();
            List<TimerMetrics> timerMetrics = keys.stream()
                    .map(key ->
                            TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                    .collect(Collectors.toList());
            String jsonStr = jacksonJsonSerializer.serialize(timerMetrics);
            return ServerMessage.newBuilder().setStr(jsonStr).build();
        }
    }
}

