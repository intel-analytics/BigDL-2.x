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


import io.grpc.*;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;


import java.io.IOException;
import java.util.UUID;
import java.util.function.Function;

public class ZooGrpcClient extends AbstractZooGrpc{
    protected static final Logger logger = Logger.getLogger(ZooGrpcClient.class.getName());
    protected String target;
    protected final String clientUUID;
    protected ManagedChannel channel;
    public ZooGrpcClient(String[] args) {
        clientUUID = UUID.randomUUID().toString();
        options = new Options();
        Option portArg = new Option(
                "t", "target", true, "URL.");
        options.addOption(portArg);
        Option configPathArg = new Option(
                "c", "config", true, "The path to config YAML file");
        options.addOption(new Option(
                "s", "service", true, "service to use"));
        options.addOption(configPathArg);
        this.args = args;


    }
    protected void parseArgs() throws IOException {
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
        services = cmd.getOptionValue("s", "").split(",");
    }
    public void loadServices() {

    }
    public void build() throws IOException {
        parseArgs();

        channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS).
                .usePlaintext()
                .build();
        loadServices();
    }
    public <I, O> O call(Function<I, O> f, I msg) {
        O r = null;
        try {
            r = f.apply(msg);
        } catch (Exception e) {
            logger.warn("failed");
        } finally {
            return r;
        }

    }
    public int t(int a) {
        return a;
    }
    public static void main(String[] args) {
        String[] a = null;
        ZooGrpcClient z = new ZooGrpcClient(a);
        Object res = z.call(z::t, 1);
        res = (Object)res;
        return;
    }
}
