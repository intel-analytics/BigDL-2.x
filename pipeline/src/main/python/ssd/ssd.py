from nn.layer import *

class Test(Model):
    """
    >>> test = Test("myworld")
    creating: createTest
    >>> print(test.value)
    hello myworld
    >>> linear = Linear(1, 2)
    creating: createLinear
    """
    def __init__(self, message, bigdl_type="float"):
        super(Test, self).__init__(None, bigdl_type, message)


def _test():
    import sys
    print sys.path
    import doctest
    from pyspark import SparkContext
    from util.common import init_engine
    from util.common import create_spark_conf
    from util.common import JavaCreator
    import ssd
    globs = ssd.__dict__.copy()
    sc = SparkContext(master="local[4]", appName="test layer",
                      conf=create_spark_conf())
    globs['sc'] = sc
    JavaCreator.set_creator_class("com.intel.analytics.bigdl.python.api.SSDPythonBigDL")  # noqa
    init_engine()
    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)

def predict(resolution, batch_size, n_partition, folder, _sc, _model, n_classes):
    return callBigDlFunc("float", "ssdPredict", resolution, batch_size, n_partition,
                         folder, _sc, _model, n_classes)

if __name__ == "__main__":
    _test()
