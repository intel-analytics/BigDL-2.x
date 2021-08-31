package com.intel.analytics.zoo.ppml;

public class FLHelper {
    String servicesList;

    // Server property
    int serverPort;

    // Client property
    String clientTarget;
    String taskID;

    public void setServicesList(String servicesList) {
        this.servicesList = servicesList;
    }

    public void setServerPort(int serverPort) {
        this.serverPort = serverPort;
    }

    public void setClientTarget(String clientTarget) {
        this.clientTarget = clientTarget;
    }

    public void setTaskID(String taskID) {
        this.taskID = taskID;
    }
}
