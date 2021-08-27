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


package com.intel.analytics.zoo.ppml.ps.test;

import com.intel.analytics.zoo.ppml.generated.FLProto.FloatTensor;
import com.intel.analytics.zoo.ppml.generated.FLProto.Table;
import com.intel.analytics.zoo.ppml.generated.FLProto.TableMetaData;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContext;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.util.InsecureTrustManagerFactory;


import javax.net.ssl.SSLException;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TestUtils {

    public static Table genRandomData(int version) {
        List<Float> dataList = new ArrayList<>();
        List<Integer> shapeList = new ArrayList<>();

        int length = 10;
        shapeList.add(length);

        for (int i = 0; i < length; i++) {
            Double temp = Math.random();
            dataList.add(temp.floatValue());
        }

        FloatTensor tensor =
                FloatTensor.newBuilder()
                        .addAllTensor(dataList)
                        .addAllShape(shapeList)
                        .build();

        TableMetaData metadata =
                TableMetaData.newBuilder()
                        .setName("first_model")
                        .setVersion(version)
                        .build();
        return Table.newBuilder()
                .putTable("first_tensor", tensor)
                .setMetaData(metadata)
                .build();
    }

    public static SslContext buildSslContext(String trustCertCollectionFilePath,
                                             String clientCertChainFilePath,
                                             String clientPrivateKeyFilePath) throws SSLException {
/*
    USAGE: certChainFilePath privateKeyFilePath [trustCertCollectionFilePath]
    Note: You only need to supply trustCertCollectionFilePath if you want to enable Mutual TLS.
    https://github.com/grpc/grpc-java/tree/master/examples/example-tls
*/
        SslContextBuilder builder = GrpcSslContexts.forClient();
        if (trustCertCollectionFilePath != null) {
            // builder.trustManager(new File(trustCertCollectionFilePath));
            // TODO Test only, need to remove during production
            builder.trustManager(InsecureTrustManagerFactory.INSTANCE);
        }
        if (clientCertChainFilePath != null && clientPrivateKeyFilePath != null) {
            builder.keyManager(new File(clientCertChainFilePath), new File(clientCertChainFilePath));
        }
        return builder.build();
    }
}
