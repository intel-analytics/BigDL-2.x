package com.intel.analytics.zoo.inference.examples.TextClassification;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import org.springframework.web.bind.annotation.RequestBody;

import org.springframework.web.bind.annotation.RequestMethod;



import java.util.concurrent.atomic.AtomicLong;

@RestController
public class GreetingController {

    private static final String template = "Hello, %s!";
    private final AtomicLong counter = new AtomicLong();

    @RequestMapping(value = "/greeting")
    public Greeting greeting(@RequestParam(value = "name", defaultValue = "World") String name) {
        return new Greeting(counter.incrementAndGet(),
                String.format(template, name));
    }

    @RequestMapping(value = "/test", method = {RequestMethod.POST})
    public Greeting testing(@RequestParam(value = "name", defaultValue = "test") String name) {
        return new Greeting(counter.incrementAndGet(),
                String.format(template, name));
    }

    @RequestMapping(value = "/predict", method = {RequestMethod.POST, RequestMethod.GET})
    public String webPredict(@RequestBody String text) {
        long begin = System.currentTimeMillis();
        if (!text.isEmpty()) {
            TextClassificationSample tc = new TextClassificationSample();
            tc.setText(text);
            String result = tc.getResult();
            long end = System.currentTimeMillis();
            return result+"#Total time elapsed " + (end - begin) + " ms.";
        } else {
            return "error,no text found";
        }

    }
}
