This page gives some instructions and tips to build and develop Analytics Zoo for Python users.

You are very welcome to add customized functionalities to Analytics Zoo to meet your own demands. 
You are also highly encouraged to contribute to Analytics Zoo so that other community users would get benefits as well.

---
## **Get Analytics Zoo Source Code**
Analytics Zoo source code is available at [GitHub](https://github.com/intel-analytics/analytics-zoo).

```bash
$ git clone https://github.com/intel-analytics/analytics-zoo.git
```

By default, `git clone` will download the development version of Analytics Zoo, if you want a release version, you can use command `git checkout` to change the version.


---
## **Package Python for pip install**
If you have modified some Python code and want to newly generate the [whl](https://pythonwheels.com/) package for pip install, you can run the following script:

```bash
bash analytics-zoo/pyzoo/dev/build.sh linux default
```

- The first argument is the __platform__ to build for. Either 'linux' or 'max'.
- The second argument is the analytics-zoo __version__ to build for. 'default' means the default version for the current branch. You can also specify a different version if you wish, for example, '0.6.0.dev1'.
- You can also add other profiles to build the package, especially Spark version and BigDL version.
For example, under the situation that `pyspark==2.4.3` is a dependency, we need to add profiles `-Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.x` to build Analytics Zoo on the same Spark version.


After running the command, you will find a `.whl` file under the folder `analytics-zoo/pyzoo/dist/`, you can then directly pip install to your local Python environment:
```bash
pip install analytics-zoo/pyzoo/dist/analytics_zoo-version-py2.py3-none-manylinux1_x86_64.whl     # for Python 2.7
pip3 install analytics-zoo/pyzoo/dist/analytics_zoo-version-py2.py3-none-manylinux1_x86_64.whl    # for Python 3.5 and Python 3.6
```

See [here](../PythonUserGuide/install/#install-from-pip-for-local-usage) for more remarks related to pip install.

See [here](../PythonUserGuide/run/#run-after-pip-install) for more instructions to run analytics-zoo after pip install.

---
