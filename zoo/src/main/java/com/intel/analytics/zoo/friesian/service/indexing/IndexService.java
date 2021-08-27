package com.intel.analytics.zoo.friesian.service.indexing;

import com.google.common.base.Preconditions;
import com.intel.analytics.zoo.faiss.swighnswlib.*;
import com.intel.analytics.zoo.faiss.utils.JniFaissInitializer;
import static com.intel.analytics.zoo.faiss.utils.IndexHelperHNSW.*;
import java.util.List;

import org.apache.log4j.Logger;

public class IndexService {
    private Index index;
    private static final Logger logger = Logger.getLogger(IndexService.class.getName());
    private static final int efConstruction = 40;
    private static final int efSearch = 256;

    IndexService(int dim) {
        Preconditions.checkArgument(JniFaissInitializer.initialized());
        index = swigfaiss.index_factory(dim, "HNSWlibInt16_32",
                MetricType.METRIC_INNER_PRODUCT);
        new ParameterSpace().set_index_parameter(index, "efSearch", efSearch);
        new ParameterSpace().set_index_parameter(index, "efConstruction", efConstruction);
        // 32768 for Int16
        new ParameterSpace().set_index_parameter(index, "scale", 32768);
    }

    public void add(int targetId, floatArray data) {
        longArray la = new longArray(1);
        la.setitem(0, targetId);
        index.add_with_ids(1, data.cast(), la.cast());
    }

    public void addWithIds(float[] data, int[] ids) {
        assert(ids.length == data.length);
        int dataNum = ids.length;
        longArray idsInput = convertIdsToLongArray(ids);
        floatArray dataInput = vectorToFloatArray(data);
        long start = System.nanoTime();
        addWithIds(dataInput, idsInput, dataNum);
        long end = System.nanoTime();
        long time = (end - start);
        logger.info("Add " + dataNum + " items to index time: " + time + " ns");
        logger.info("Current NTotal: " + this.getNTotal());
    }

    public void addWithIds(floatArray data, longArray ids, int dataNum) {
        index.add_with_ids(dataNum, data.cast(), ids.cast());
    }
    public void save(String path) {
        swigfaiss.write_index(this.index, path);
    }

    public void load(String path) {
        this.index = swigfaiss.read_index(path);
        new ParameterSpace().set_index_parameter(this.index, "efSearch", efSearch);
        // 32768 for Int16
        new ParameterSpace().set_index_parameter(this.index, "scale", 32768);
    }

    int getNTotal() {
        return this.index.getNtotal();
    }

    public boolean isTrained() {
        return this.index.getIs_trained();
    }

    public void train(int dataSize, floatArray xb) {
        logger.info("Start training");
        long start = System.nanoTime();
        this.index.train(dataSize, xb.cast());
        long end = System.nanoTime();
        long time = (end - start);
        logger.info("Training time: " + time + " ns");
    }

    public int[] search(floatArray query, int k) {
        longArray I = new longArray(k);
        floatArray D = new floatArray(k);
        index.search(1, query.cast(), k, D.cast(), I.cast());
        logger.info(show(I, 1, k));
        logger.info(show(D, 1, k));
        int[] candidates = new int[k];
        for (int i = 0; i < k; i ++) {
            candidates[i] = I.getitem(i);
        }
        return candidates;
    }

    public static floatArray vectorToFloatArray(float[] vector) {
        int d = vector.length;
        floatArray fa = new floatArray(d);
        for (int j = 0; j < d; j++) {
            fa.setitem(j, vector[j]);
        }
        return fa;
    }

    public static floatArray listOfVectorToFloatArray(List<float[]> vectors) {
        int nb = vectors.size();
        int d = vectors.get(0).length;
        floatArray fa = new floatArray(d * nb);
        for (int i = 0; i < nb; i++) {
            float[] vector = vectors.get(i);
            for (int j = 0; j < d; j++) {
                fa.setitem(d * i + j, vector[j]);
            }
        }
        return fa;
    }

    public static longArray convertIdsToLongArray(List<Integer> ids) {
        int[] idArr = ids.stream().mapToInt(i -> i).toArray();
        return convertIdsToLongArray(idArr);
    }

    public static longArray convertIdsToLongArray(int[] ids) {
        int num = ids.length;
        longArray la = new longArray(num);
        for (int i = 0; i < num; i ++) {
            la.setitem(i, ids[i]);
        }
        return la;
    }
}
