# Colab User Guide

---

You can use Analytics Zoo without any installation by using  [Google Colab](https://colab.research.google.com/). Analytics Zoo also include a collection of [notebooks]() that can be directly opened and run in Colab.

### **1. Open a Colab Notebook**

<TODO: add instructions for opening notebooks in `docs/colab-notebook/`>

### **2. Notebook Setup**

The first few cells of the notebook contains the code necessary to set up Analytics Zoo and other libraries.

**Install Java 8**

Run the following command on the Google Colab to install jdk 1.8

```bash
# Install jdk8
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
# Set jdk environment path which enables you to run Pyspark in your Colab environment.
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
!update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
```

**Install Conda**

If the application requires Conda, you can execute the code bellow on Colab to install Conda.

```bash
# Install Miniconda
!wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
!chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
!./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local

# Update Conda
!conda install --channel defaults conda python=3.6 --yes
!conda update --channel defaults --all --yes

# Append to the sys.path
import sys
_ = (sys.path
        .append("/usr/local/lib/python3.6/site-packages"))

os.environ['PYTHONHOME']="/usr/local"
```

**Install Analytics Zoo**

You can install the latest pre-release version using `pip install --pre analytics-zoo`. 
```bash
# Install latest pre-release version of Analytics Zoo 
# Installing Analytics Zoo from pip will automatically install pyspark, bigdl, and their dependencies.
!pip install --pre analytics-zoo
```

**Install Python Dependencies**

As Colab python environment provides some built-in Python libraries, you should check if the library versions are compatible with your application. You may refer [compatibility](https://analytics-zoo.readthedocs.io/en/latest/doc/UserGuide/python.html#compatibility) to specify the python library version that Analytics Zoo supports.
