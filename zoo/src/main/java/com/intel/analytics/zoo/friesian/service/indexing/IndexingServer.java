package com.intel.analytics.zoo.friesian.service.indexing;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.nn.abstractnn.Activity;
import com.intel.analytics.zoo.faiss.swighnswlib.floatArray;
import com.intel.analytics.zoo.friesian.generated.feature.FeatureServiceGrpc;
import com.intel.analytics.zoo.friesian.generated.feature.Features;
import com.intel.analytics.zoo.friesian.generated.feature.IDs;
import com.intel.analytics.zoo.friesian.generated.indexing.*;
import com.intel.analytics.zoo.serving.http.JacksonJsonSerializer;
import io.grpc.*;
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.util.*;

import com.intel.analytics.zoo.pipeline.inference.InferenceModel;

import java.util.concurrent.TimeUnit;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import java.util.stream.Collectors;

import org.apache.commons.cli.*;
import com.intel.analytics.zoo.friesian.utils.ConfigParser;
import com.intel.analytics.zoo.friesian.utils.TimerMetrics;
import com.intel.analytics.zoo.friesian.utils.TimerMetrics$;
import com.intel.analytics.zoo.friesian.utils.feature.FeatureUtils;
import com.intel.analytics.zoo.friesian.utils.vectorsearch.IndexUtils;

public class IndexingServer {
    private static final Logger logger = Logger.getLogger(IndexingServer.class.getName());

    private final int port;
    private final Server server;

    /** Create a Indexing server listening on {@code port} using {@code featureFile} database. */
    public IndexingServer(int port, String configPath) {
        this(ServerBuilder.forPort(port), port, configPath);
    }

    /** Create a RouteGuide server using serverBuilder as a base and features as data. */
    public IndexingServer(ServerBuilder<?> serverBuilder, int port, String configPath) {
        this.port = port;
        Logger.getLogger("org.apache").setLevel(Level.ERROR);
        server = serverBuilder.addService(new IndexingService(configPath)).build();
    }

    /** Start serving requests. */
    public void start() throws IOException {
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            // Use stderr here since the logger may have been reset by its JVM shutdown hook.
            System.err.println("*** shutting down gRPC server since JVM is shutting down");
            try {
                IndexingServer.this.stop();
            } catch (InterruptedException e) {
                e.printStackTrace(System.err);
            }
            System.err.println("*** server shut down");
        }));
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
        int port = Integer.parseInt(cmd.getOptionValue("port", "8084"));
        String configPath = cmd.getOptionValue("config", "config.yaml");
        IndexingServer server = new IndexingServer(port, configPath);
        Logger.getLogger("org.apache").setLevel(Level.ERROR);
        server.start();
        server.blockUntilShutdown();
    }

    private static class IndexingService extends IndexingGrpc.IndexingImplBase {
        private InferenceModel userModel;
        private InferenceModel itemModel;
        private IndexService indexService;
        private boolean callFeatureService = false;
        private FeatureServiceGrpc.FeatureServiceBlockingStub featureServiceStub;
        MetricRegistry metrics = new MetricRegistry();
        Timer overallTimer = metrics.timer("indexing.overall");
        Timer predictTimer = metrics.timer("indexing.predict");
        Timer faissTimer = metrics.timer("indexing.faiss");

        IndexingService(String configPath) {
            ConfigParser parser = new ConfigParser(configPath);
            IndexUtils.helper_$eq(parser.loadConfig());
            if (IndexUtils.helper().getGetFeatureFromFeatureService()) {
                callFeatureService = true;
                ManagedChannel featureServiceChannel =
                        ManagedChannelBuilder.forTarget(IndexUtils.helper().getFeatureServiceURL())
                                .usePlaintext().build();
                featureServiceStub = FeatureServiceGrpc.newBlockingStub(featureServiceChannel);
            } else {
                userModel = IndexUtils.helper()
                        .loadInferenceModel(IndexUtils.helper().getModelParallelism(),
                                IndexUtils.helper().getUserModelPath(), null);
            }
            // load or build faiss index
            indexService = new IndexService(128);
            if (IndexUtils.helper().loadSavedIndex()) {
                assert(IndexUtils.helper().getIndexPath() != null): "indexPath must be provided " +
                        "if loadSavedIndex=true.";
                indexService.load(IndexUtils.helper().getIndexPath());
            } else {
                if (IndexUtils.helper().getItemModelPath() != null) {
                    itemModel = IndexUtils.helper()
                            .loadInferenceModel(IndexUtils.helper().getModelParallelism(),
                                    IndexUtils.helper().getItemModelPath(), null);
                    IndexUtils.helper().setItemModel(itemModel);
                }
                String dataDir = IndexUtils.helper().getInitialDataPath();
                IndexUtils.loadItemData(indexService, dataDir, itemModel,1000000);
                assert(this.indexService.isTrained());
            }
            System.out.printf("Index service nTotal = %d\n", this.indexService.getNTotal());
        }

        @Override
        public void searchCandidates(Query request,
                                     StreamObserver<Candidates> responseObserver) {
            Candidates candidates;
            try {
                candidates = search(request);
            } catch (Exception e) {
                e.printStackTrace();
                logger.warn(e.getMessage());
                responseObserver.onError(Status.INTERNAL.withDescription(e.getMessage())
                        .asRuntimeException());
                return;
            }
            responseObserver.onNext(candidates);
            responseObserver.onCompleted();
        }

        @Override
        public void addItem(Item request,
                            StreamObserver<Empty> responseObserver) {
            responseObserver.onNext(addItemToIndex(request));
            responseObserver.onCompleted();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<com.intel.analytics.zoo.friesian.generated.indexing.ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
        }

        @Override
        public void resetMetrics(Empty request, StreamObserver<Empty> responseObserver) {
            metrics = new MetricRegistry();
            overallTimer = metrics.timer("indexing.overall");
            predictTimer = metrics.timer("indexing.predict");
            faissTimer = metrics.timer("indexing.faiss");
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }

        private Candidates search(Query msg) throws Exception {
            Timer.Context overallContext = overallTimer.time();
            int userId = msg.getUserID();
            int k = msg.getK();
            Timer.Context predictContext = predictTimer.time();
            Activity userFeature;
            if (callFeatureService) {
                IDs userIds = IDs.newBuilder().addID(userId).build();
                Features feature = featureServiceStub.getUserFeatures(userIds);
                Object[] activityList =
                        Arrays.stream(FeatureUtils.featuresToObject(feature))
                                .filter(Objects::nonNull).toArray();
                if (activityList.length == 0) {
                    throw new Exception("Can't get user feature from feature service");
                }
                userFeature = (Activity) activityList[0];
            } else {
                userFeature = this.userModel
                        .doPredict(IndexUtils.constructActivity(Collections.singletonList(userId)));
            }
            predictContext.stop();
            Timer.Context faissContext = faissTimer.time();
            float[] userFeatureList = IndexUtils.activityToFloatArr(userFeature);
            int[] candidates =
                    indexService.search(IndexService.vectorToFloatArray(userFeatureList), k);
            faissContext.stop();
            Candidates.Builder result = Candidates.newBuilder();
            // TODO: length < k
            for (int i = 0; i < k; i ++) {
                result.addCandidate(candidates[i]);
            }
            overallContext.stop();
            return result.build();
        }

        private Empty addItemToIndex(Item msg) {
            // TODO: multi server synchronize
            System.out.printf("Index service nTotal before = %d\n", this.indexService.getNTotal());
            int itemId = msg.getItemID();
            Activity itemFeature = predict(this.itemModel,
                    IndexUtils.constructActivity(Collections.singletonList(itemId)));
            float[] itemFeatureList = IndexUtils.activityToFloatArr(itemFeature);
            addToIndex(itemId, itemFeatureList);
            System.out.printf("Index service nTotal after = %d\n", this.indexService.getNTotal());
            return Empty.newBuilder().build();
        }

        private Activity predict(InferenceModel inferenceModel, Activity data){
            return inferenceModel.doPredict(data);
        }

        private void addToIndex(int targetId, float[] vector) {
            floatArray fa = IndexService.vectorToFloatArray(vector);
            this.indexService.add(targetId, fa);
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
