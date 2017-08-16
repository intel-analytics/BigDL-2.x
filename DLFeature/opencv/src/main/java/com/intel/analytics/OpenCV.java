package com.intel.analytics;

import java.io.*;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

/**
 * OpenCV Library Wrapper for JVM
 */
public class OpenCV {
    private static boolean isLoaded = false;
    private static File tmpFile = null;

    public static void load() {
        try {
            String jopencvFileName = "opencv/linux/x86_64/libopencv_java320.so";
            if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                jopencvFileName = "opencv/osx/x86_64/libopencv_java320.dylib";
            }
            tmpFile = extract(jopencvFileName);
            System.load(tmpFile.getAbsolutePath());
            tmpFile.delete(); // delete so temp file after loaded
            isLoaded = true;

        } catch (Exception e) {
            isLoaded = false;
            e.printStackTrace();
            // TODO: Add an argument for user, continuing to run even if opencv load failed.
            throw new RuntimeException("Failed to load OpenCV");
        }
    }

    /**
     * Check if opencv is loaded
     * @return
     */
    public static boolean isOpenCVLoaded() {
        return isLoaded;
    }

    /**
     * Get the temp path of the .so file
     * @return
     */
    public static String getTmpSoFilePath() {
        if(tmpFile == null)
            return "";
        else
            return tmpFile.getAbsolutePath();
    }

    // Extract so file from jar to a temp path
    private static File extract(String path) {
        try {
            URL url = OpenCV.class.getClassLoader().getResource(path);
            if (url == null) {
                throw new Error("Can't find so file in jar, path = " + path);
            }

            InputStream in = OpenCV.class.getClassLoader().getResourceAsStream(path);
            File file = createTempFile("dlNativeLoader", path.substring(path.lastIndexOf("/") + 1));

            ReadableByteChannel src = newChannel(in);
            FileChannel dest = new FileOutputStream(file).getChannel();
            dest.transferFrom(src, 0, Long.MAX_VALUE);
            return file;
        } catch (Throwable e) {
            throw new Error("Can't extract so file to /tmp dir");
        }
    }

    public static void main(String[] args) {
        OpenCV openCV = new OpenCV();
        openCV.load();
    }

}
