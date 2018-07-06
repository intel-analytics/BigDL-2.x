
package com.intel.analytics.zoo.inference.examples.preprocessor;


import com.intel.analytics.zoo.pipeline.inference.JTensor;
import com.intel.analytics.zoo.preprocess.ITextProcessing;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;

import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Scanner;
import java.io.IOException;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

public class GloveTextProcessing extends ITextProcessing {

	@Override
	public Map<String, List<Float>> loadEmbedding(String embDir) {
		Map<String, List<Float>> embMap = new HashMap<>();
		String filename = embDir + "/glove.6B.200d.txt";
		try(BufferedReader br = new BufferedReader(new FileReader(filename))) {
			for(String line; (line = br.readLine()) != null; ) {
				String[] parts = line.split(" ", 2);
				String word = parts[0];
				String emb = parts[1];
				Scanner scanner = new Scanner(emb);
				List<Float> list = new ArrayList<>();
				while (scanner.hasNextFloat()) {
					list.add(scanner.nextFloat());
				}
				embMap.put(word, list);
			}
		}catch (IOException e) {
			e.printStackTrace();
		}
		return embMap;
	}

	// @Override
//	public JTensor preprocess(String text) {
//		List<String> tokens = tokenize(text);
//		List<String> shapedTokens = shaping(stopWords(tokens,1),500);
//		Map<String, List<Float>> embMap = loadEmbedding(System.getProperty("EMBEDDING_PATH", "/home/yidiyang/workspace/dataset/glove.6B"));
//		List<List<Float>> vectorizedTokens = vectorize(shapedTokens, embMap);
//		List<Float> data = Lists.newArrayList(Iterables.concat(vectorizedTokens));
//		List<Integer> shape = new ArrayList<>();
//		shape.add(vectorizedTokens.size());
//		shape.add(vectorizedTokens.get(0).size());
//		JTensor tensorInput = new JTensor(data, shape);
//		return tensorInput;
//	}
}



