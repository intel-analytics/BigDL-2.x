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

import com.intel.analytics.zoo.grpc.ZooGrpcServer;
import com.intel.analytics.zoo.ppml.psi.PSIServiceImpl;
import com.intel.analytics.zoo.ppml.psi.test.BenchmarkClient;
import io.grpc.BindableService;
import io.grpc.ServerBuilder;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NettyServerBuilder;
import io.netty.handler.ssl.ClientAuth;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.SSLException;
import java.io.File;
import java.io.IOException;

/**
 * FLServer is Analytics Zoo PPML gRPC server used for FL based on ZooGrpcServer
 * It would parse ZooGrpcServer args and add its specific args after it.
 * User could also call main method and parse server type to start gRPC service
 * Supported types: PSI, VFL,
 */
public class FLServer extends ZooGrpcServer {
    private static final Logger logger = LoggerFactory.getLogger(FLServer.class);
    String certChainFilePath;
    String privateKeyFilePath;
    String trustCertCollectionFilePath;
    FLServer(String[] args, BindableService service) {
        super(args, service);
        options.addOption(new Option(
                "cc", "certChainFilePath", true, "certChainFilePath"));
        options.addOption(new Option(
                "pk", "privateKeyFilePath", true, "privateKeyFilePath"));
        options.addOption(new Option(
                "tcc", "trustCertCollectionFilePath", true, "trustCertCollectionFilePath"));
    }
    FLServer(String[] args) {
        this(args, null);
    }
    void startWithTls() throws IOException {
        // TODO: to add for multi-services
        parseArgs();
        certChainFilePath = cmd.getOptionValue("cc", null);
        privateKeyFilePath = cmd.getOptionValue("pk", null);
        trustCertCollectionFilePath = cmd.getOptionValue("tcc", null);
        NettyServerBuilder serverBuilder = NettyServerBuilder.forPort(port)
                .addService(service);
        if (certChainFilePath != null && privateKeyFilePath != null) {
            serverBuilder.sslContext(getSslContext());
        }
    }

    SslContext getSslContext() throws SSLException {
        SslContextBuilder sslClientContextBuilder = SslContextBuilder.forServer(new File(certChainFilePath),
                new File(privateKeyFilePath));
        if (trustCertCollectionFilePath != null) {
            sslClientContextBuilder.trustManager(new File(trustCertCollectionFilePath));
            sslClientContextBuilder.clientAuth(ClientAuth.REQUIRE);
        }
        return GrpcSslContexts.configure(sslClientContextBuilder).build();
    }

    @Override
    public void parseArgs() throws IOException {
        FLHelper flHelper = getCmd(FLHelper.class);
        if (flHelper != null) {
            serviceList = flHelper.servicesList;
            port = flHelper.serverPort;
        }

        super.parseArgs();
    }

    @Override
    public void build() throws IOException {
        parseArgs();
        ServerBuilder builder = ServerBuilder.forPort(port);
        if (service != null) {
            builder.addService(service);
        }
        for (String service : services) {
            if (service.equals("psi")) {
                builder.addService(new PSIServiceImpl());
            } else if (service.equals("ps")) {
                // TODO: add algorithms here
            } else {
                logger.warn("Type is not supported, skipped. Type: " + service);
            }
        }
        server = builder.maxInboundMessageSize(Integer.MAX_VALUE).build();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        FLServer flServer = new FLServer(args);
        flServer.build();
        flServer.start();
        flServer.blockUntilShutdown();
    }
}
