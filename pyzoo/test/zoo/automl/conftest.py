from zoo import init_spark_on_local
from zoo.ray.util.raycontext import RayContext
import pytest
sc = None
ray_ctx = None

@pytest.fixture(autouse=True, scope='session')
def automl_fixture():
    sc = init_spark_on_local(cores=4, spark_log_level="INFO")
    ray_ctx = RayContext(sc=sc)
    ray_ctx.init()
    yield
    ray_ctx.stop()
    sc.stop()

@pytest.fixture()
def setUpModule():
    sc = init_spark_on_local(cores=4, spark_log_level="INFO")
    ray_ctx = RayContext(sc=sc)
    ray_ctx.init()


def tearDownModule():
    ray_ctx.stop()
    sc.stop()