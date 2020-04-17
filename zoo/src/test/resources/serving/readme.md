# Cluster Serving Benchmark and Correctness Test

## Benchmark
Benchmark test is to test the benchmark result of backend max throuphput, end-to-end single-stream average latency, end-to-end multi-stream 99% latency.

Image enqueue contains 4 phases, running in sequential order

1. 1 thread enqueue 1 image, and wait for response, to warm up Cluster Serving, aka. model, tensor initialization.
2. 1 thread enqueue 1 image, and wait for response, repeat for 100 times, to test end-to-end single-stream average latency.
3. 100 threads, each enqueue 1 image, and wait for response, repeat for 10 times, to test end-to-end multi-stream 99% latency.
4. 1 thread enqueue 5000 images at once, to test backend max throuput.

To test benchmark, run `benchmark.sh`.

## Correctness
Correctness test is to test Top-1 accuracy, Top-5 accuracy, and no data lost in pipeline.

To test correctness, run `imagenet_correctness.sh`, note that you need to provide image directory path, and label txt file in the same position and filename, and the only file type supported is JPEG. e.g.
```
--path
  |-- data
     |-- img1.jpeg
     |-- img2.jpeg
  |-- data.txt   // label text file
```
