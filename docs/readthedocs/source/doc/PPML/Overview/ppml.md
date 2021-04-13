# PPML (Privacy Preserving Machine Learning)

Analytics-Zoo provides an end-to-end PPML platform for Big Data AI.

## PPML for Big Data AI

To take full advantage of the value of big data, especially the value of private or sensitive data, customers need to build a trusted platform under the guidance of privacy laws or regulation, such as [GDPR](https://gdpr-info.eu/). This requirement raises a big challenge to customers who already have big data and big data applications, such as Spark/SparkSQL, Flink and AI applications. Migrating these applications into privacy preserving way requires lots of additional efforts.

To reslove this problem, Analytics-Zoo chooses [Intel SGX (Software Guard Extensions)](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html), a widely used [TEE (Trusted Execution Environment)](https://en.wikipedia.org/wiki/Trusted_execution_environment) technology, as main security building block for this PPML platforms. Different from other PPML technologies, e.g., [HE (Homomorphic Encryption)](https://en.wikipedia.org/wiki/Homomorphic_encryption), [MPC (Multi-Party Computation) or SMC (Secure Multi-Party Computation)](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and [DP (Differential Privacy)](https://en.wikipedia.org/wiki/Differential_privacy), Intel SGX performs well on all measures (security, performance and utility).

![PPML Technologies](../../../../../docs/Image/PPML/ppml_tech.png)

Based on Intel SGX (Software Guard Extensions) and LibOS projects ([Graphene](https://grapheneproject.io/) and [Occlum](https://occlum.io/)), Analytics-Zoo empowers our customers (e.g., data scientists and big data developers) to build PPML applications on top of large scale dataset without impacting existing applications.

![PPML Architecture](../../../../../docs/Image/PPML/ppml_arch.png#center)

Note: Intel SGX requires hardware support, please [check if your CPU has this feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). In [3rd Gen Intel Xeon Scalable Processors](https://newsroom.intel.com/press-kits/3rd-gen-intel-xeon-scalable/), SGX allows up to 1TB of data to be included in secure enclaves.

**Key features**

- Protecting data and model confidentiality
- Trusted big data & AI Platform based on Intel SGX

**Scenario**

- Protecting sensitive input/output data (computation, training and inference) in big data applications, e.g.,data analysis or machine learning on healthcare dataset
- Protecting propretary model in training and inference, e.g., secured model inference with self-owned model

## Trusted Big Data Analytics and ML



**Scenario**

- Batch computation/analytics on senstive data, e.g., privacy preserved Spark jobs on sensitive data
- Interactive computation/analytics on sensitive data, e.g., privacy preserved SparkSQL on sensitive data
- Distributed machine learning & deep Learning on sensitive data

**Get started**

- Env setup (DockFIle)
- Spark example (pi)
- SparkSQL
- TPC-H
- BigDL Training

```bash
```

## Trusted Realtime Compute and ML

**Scenario**

- Real time data computation/analytics on sensitive data, e.g., privacy preserved Flink jobs on sensitive data
- Privacy preserved distributed model inference with propretary model on sensitive data

**Get started**
- Env setup (DockFIle)
- Flink example (word count)
- Cluster serving

```bash
```
