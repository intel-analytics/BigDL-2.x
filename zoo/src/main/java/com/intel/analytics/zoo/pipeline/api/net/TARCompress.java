package com.intel.analytics.zoo.pipeline.api.net;

import java.io.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class TARCompress {

    public static void unzip(InputStream inputStream, File outputPath) throws IOException {
        try {
            byte[] buffer = new byte[2048];
            ZipInputStream istream = new ZipInputStream(inputStream);
            ZipEntry entry;
            while ((entry = istream.getNextEntry()) != null) {
                File entryDestination = new File(outputPath, entry.getName());
                if (entry.isDirectory()) {
                    entryDestination.mkdirs();
                } else {
                    entryDestination.getParentFile().mkdirs();
                    FileOutputStream fos = new FileOutputStream(entryDestination);
                    BufferedOutputStream bos = new BufferedOutputStream(fos, buffer.length);

                    int len;
                    while ((len = istream.read(buffer)) > 0) {
                        bos.write(buffer, 0, len);
                    }
                    bos.close();
                }
            }
        } catch (IOException io) {
            System.out.println("error during loading loading pytorch libs");
            throw io;
        }
    }
}
