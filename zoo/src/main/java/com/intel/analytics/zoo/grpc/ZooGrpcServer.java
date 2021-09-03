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


package com.intel.analytics.zoo.grpc;

import io.grpc.BindableService;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.LinkedList;
import java.util.concurrent.TimeUnit;

/**
 * All Analytics Zoo gRPC server are based on ZooGrpcServer
 * This class could also be directly used for start a single service
 */
public class ZooGrpcServer extends AbstractZooGrpc{
    protected static final Logger logger = Logger.getLogger(ZooGrpcServer.class.getName());
    protected int port;
    protected Server server;
    protected LinkedList<BindableService> serverServices;


    /**
     * One Server could support multiple servives.
     * Also support a single service constructor
     * @param service
     */
    public ZooGrpcServer(BindableService service) {
        this(null, service);
    }
    public ZooGrpcServer(String[] args, BindableService service) {
        serverServices = new LinkedList<>();
        if (service != null) {
            serverServices.add(service);
        }
        this.args = args;

    }
    public ZooGrpcServer(String[] args) {
        this(args, null);
    }

    public void parseArgs() throws IOException {}


    /** Entrypoint of ZooGrpcServer */
    public void build() throws IOException {
        parseArgs();
        ServerBuilder builder = ServerBuilder.forPort(port);
        for (BindableService bindableService : serverServices) {
            builder.addService(bindableService);
        }
        server = builder.maxInboundMessageSize(Integer.MAX_VALUE).build();
    }

    /** Start serving requests. */
    public void start() throws IOException {
        /* The port on which the server should run */
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                try {
                    ZooGrpcServer.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
                System.err.println("*** server shut down");
            }
        });
    }
    public void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }
    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    public void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }
}
