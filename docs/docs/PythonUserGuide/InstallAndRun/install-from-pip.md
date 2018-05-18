## **NOTES**

- Pip install supports __Mac__ and __Linux__ platforms.
- You need to install Java __>= JDK8__ before running Analytics Zoo, which is required by __pyspark__.
- Pip install only supports __local__ mode. Cluster mode might be supported in the future. For those who want to use Analytics Zoo in cluster mode, please try to [install without pip](./install-without-pip.md).
- We've tested this package with __Python 2.7__, __Python 3.5__ and __Python 3.6__. Only these three Python versions are supported for now.
- If you use pip install, it is __not__ necessary to set `SPARK_HOME`.

---
## **Install Analytics-Zoo-0.1.0.dev0**

Install Analytics Zoo release via pip (we tested this on pip 9.0.1)

**Remark:**

- You might need to add `sudo` if without permission for the installation.

- `bigdl==0.5.0` and its dependencies (including `pyspark`, `numpy` and `six`) will be automatically installed first before installing Analytics Zoo if they haven't been detected locally.
```bash
pip install --upgrade pip
pip install analytics-zoo==0.1.0.dev0     # for Python 2.7
pip3 install analytics-zoo==0.1.0.dev0    # for Python 3.5 and Python 3.6
```
