##Analytis Zoo Examples

### Notebooks:

| notebook | description | 
| --------- | ------------- |
| [Anomaly Detection](../../apps/anomaly-detection) | Unsupervised anomaly detection using Analytics Zoo Keras-Style API. |
| [Image Transfer Learning](../../apps/dogs-vs-cats) | Trains a dogs-vs-cats model from Inception-V1 model, using the convenient transfer learning API and with Spark DataFrame. | 
| [Fraud Detection](../../apps/fraud-detection) | Scala notebook to train a fraud detection model with public dataset, using Spark DataFrame and ML pipeline. |
| [image-augmentation](../../apps/image-augmentation) | Demonstrates the image argumentation and preprocessing provided by Analytics Zoo. |
| [image-augmentation-3d](../../apps/image-augmentation-3d) | Demonstrates the 3D image argumentation and preprocessing provided by Analytics Zoo. | |
| [image-similarity](../../apps/image-similarity) | Use Real Estate images as an example, extract semantic tags and image embeddings to support image search and recommendation |
| [object-detection](../../apps/object-detection) | Demonstrates the pre-trained object detection model in Analytics Zoo|
| [recommendation-ncf](../../apps/recommendation-ncf) | Trains a Neural Collaborative Filtering model with movie-lens data |
| [recommendation-wide-n-deep](../../apps/recommendation-wide-n-deep) |Trains a Wide and Deep model with movie-lens data |
| [sentiment-analysis](../../apps/sentiment-analysis) | Trains multiple models (CNN, LSTM, GRU, Bi-LSTM, CNN-LSTM etc.) for sentiment classification task. |
| [tfnet](../../apps/tfnet) | Inference with the pre-trained Tensorflow model on Spark with the TFNet API. |
| [variational-autoencoder](../../apps/variational-autoencoder) | Contains notebooks to generate digits and human faces with variational autoencoder |
| [web-service-sample](../../apps/web-service-sample) | Serves the Analytics Zoo model (text classification and recommendation) in a web service |

### Examples

| example | description | 
| --------- | ------------- |
| Anomaly Detection [Scala](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/anomalydetection) [Python](../../pyzoo/zoo/examples/anomalydetection) | Unsupervised anomaly detection using Analytics Zoo high level AnomalyDetector API |
| Chatbot [Scala](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/chatbot) | Demonstrates how to train a chatbot and use it to inference answers for queries |
| Image Classification [Scala](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/imageclassification) [Python](../../pyzoo/zoo/examples/imageclassification) | Image classification using Analytics Zoo high level ImageClassifier API |
| Object Detection [Scala](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/objectdetection) [Python](../../pyzoo/zoo/examples/objectdetection) | Detects objects in image with pre-trained model and high level object detection API |
| Text Classification [Scala](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification) [Python](../../pyzoo/zoo/examples/textclassification) | Uses pre-trained GloVe embeddings to convert words to vectors, and trains a CNN, LSTM or GRU `TextClassifier` model on 20 Newsgroup dataset. |
| TensorFlow Support [Scala](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/tfnet) [Python](../../pyzoo/zoo/examples/tensorflow) | Use tensorflow API to define model and run training or inference with Analytics Zoo on Apache Spark |
| DataFrame and ML pipeline [Scala](../../zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes) [Python](../../pyzoo/zoo/examples/nnframes) | Use DataFrame-based API to train or inference deep learning models, compatible with Spark ML pipeline |






