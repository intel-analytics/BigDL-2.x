package com.intel.analytics.zoo.grpc;


import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.UUID;

public class ZooGrpcClient {
    protected static final Logger logger = LoggerFactory.getLogger(ZooGrpcClient.class);
    protected String[] args;
    protected Options options;
    protected String target;
    protected String configPath;
    protected final String clientUUID;
    protected CommandLine cmd;
    protected ManagedChannel channel;
    public ZooGrpcClient(String[] args) {
        clientUUID = UUID.randomUUID().toString();
        options = new Options();
        Option portArg = new Option(
                "t", "target", true, "URL.");
        options.addOption(portArg);
        Option configPathArg = new Option(
                "c", "config", true, "The path to config YAML file");
        options.addOption(configPathArg);
        this.args = args;


    }
    protected void parseArgs() {
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
        }
        assert cmd != null;
        target = cmd.getOptionValue("t", "locaohost:8980");
        configPath = cmd.getOptionValue("config", "config.yaml");
    }
    public void build() {
        parseArgs();
        channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS).
                .usePlaintext()
                .build();
    }

}
