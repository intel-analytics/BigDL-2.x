package com.intel.analytics.zoo.grpc.azinference;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.dllib.grpc.ZooGrpcServer;
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity;
import com.intel.analytics.bigdl.dllib.tensor.Tensor;
import com.intel.analytics.bigdl.dllib.utils.grpc.JacksonJsonSerializer;
import com.intel.analytics.bigdl.orca.inference.InferenceModel;
import com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceGrpc;
import com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Content;
import com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.Prediction;
import com.intel.analytics.zoo.grpc.generated.azinference.AZInferenceProto.ServerMessage;
import com.intel.analytics.zoo.grpc.utils.EncodeUtils;
import io.grpc.ServerInterceptors;
import io.grpc.stub.StreamObserver;
import io.prometheus.client.exporter.HTTPServer;
import me.dinowernli.grpc.prometheus.Configuration;
import me.dinowernli.grpc.prometheus.MonitoringServerInterceptor;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import utils.TimerMetrics;
import utils.TimerMetrics$;
import utils.Utils;
import utils.gRPCHelper;

import java.io.IOException;
import java.util.Base64;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;


public class AZInferenceServer extends ZooGrpcServer {
    private static final Logger logger = Logger.getLogger(AZInferenceServer.class.getName());

    /**
     * One Server could support multiple services.
     *
     * @param args
     */
    public AZInferenceServer(String[] args) {
        super(args);
        configPath = "config_inference.yaml";
        Logger.getLogger("org").setLevel(Level.ERROR);
    }

    @Override
    public void parseConfig() throws IOException {
        Utils.helper_$eq(getConfigFromYaml(gRPCHelper.class, configPath));
        if (Utils.helper() != null) {
            port = Utils.helper().getServicePort();
        }

        if (Utils.runMonitor()) {
            logger.info("Starting monitoringInterceptor....");
            MonitoringServerInterceptor monitoringInterceptor =
                    MonitoringServerInterceptor.create(Configuration.allMetrics()
                            .withLatencyBuckets(Utils.getPromBuckets()));
            serverDefinitionServices.add(ServerInterceptors
                    .intercept(new AZInferenceService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new AZInferenceService());
        }
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        AZInferenceServer azInferenceServer = new AZInferenceServer(args);
        azInferenceServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        azInferenceServer.start();
        azInferenceServer.blockUntilShutdown();
    }

    private static class AZInferenceService extends AZInferenceGrpc.AZInferenceImplBase {
        private final InferenceModel model;
        private int cnt = 0;
        private long time = 0;
        private int pre = 1000;
        private MetricRegistry metrics = new MetricRegistry();
        Timer overallTimer = metrics.timer("inference.overall");

        AZInferenceService() {
            gRPCHelper helper = Utils.helper();
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
            Timer.Context overallContext = overallTimer.time();
            long start = System.nanoTime();
            String encodedStr = msg.getEncodedStr();
            byte[] bytes1 = Base64.getDecoder().decode(encodedStr);
            Activity input = (Activity) EncodeUtils.bytesToObj(bytes1);
            Activity predictResult = model.doPredict(input);
            String res = Utils.tensorToNdArrayString((Tensor<Object>)predictResult);
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
            overallContext.stop();
            return Prediction.newBuilder().setPredictStr(res).build();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
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

        @Override
        public void resetMetrics(Empty request, StreamObserver<Empty> responseObserver) {
            metrics = new MetricRegistry();
            overallTimer = metrics.timer("inference.overall");
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }
    }
}
