# Summary

With colaboratory, we can easily setup and run code in the cloud. How can we run Analytics Zoo [tutorials](https://github.com/intel-analytics/zoo-tutorials) or examples on colaboratory? 

First, load or create the notebook file in colaboratory. Then, prepare the environment. Only you need to prepare is installing JDK and Analytics Zoo. As installing analytics-zoo from pip will automatically install pyspark, you are recommended to not install pyspark anymore.  

## Prepare Environment

### Install Java 8

Run the command on the colaboratory file to install jdk 1.8:

```python
# Install jdk8
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# Set jdk environment path which enables you to run Pyspark in your Colab environment.
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

### **Install Analytics Zoo from pip**

You can add the following command on your colab file to install the analytics-zoo via pip easily:

```python
# Install latest release version of Analytics-Zoo 
# Installing analytics-zoo from pip will automatically install pyspark. 
!pip install analytics-zoo
```

## Run Github Notebook on colaboratory

If you would like to open the Notebook in a GitHub repo directly, the only thing you need to do is:

- Open the Notebook file on GitHub in a browser (So the URL ends in *.ipynb*).

- Change the URL from https://github.com/full_path_to_the_notebook to https://colab.research.google.com/github/full_path_to_the_notebook

  For example, change URL of tutorial https://github.com/intel-analytics/zoo-tutorials/blob/master/keras/2.1-a-first-look-at-a-neural-network.ipynb to https://colab.research.google.com/github/intel-analytics/zoo-tutorials/blob/master/keras/2.1-a-first-look-at-a-neural-network.ipynb 

Then, prepare environment of Java8 and Analytics Zoo as instructions above. If the notebook is not authored, you could make a copy and run it within the instructions.
