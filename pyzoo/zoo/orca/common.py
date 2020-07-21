#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


class OrcaContextMeta(type):

    _orca_eager_mode = True
    _orca_pandas_read_backend = "spark"

    @property
    def orca_eager_mode(cls):
        """
        Whether to compute eagerly for SparkXShards.
        Default to be True.
        """
        return cls._orca_eager_mode

    @orca_eager_mode.setter
    def orca_eager_mode(cls, value):
        assert isinstance(value, bool), "orca_eager_mode should either be True or False"
        cls._orca_eager_mode = value

    @property
    def orca_pandas_read_backend(cls):
        """
        The backend for reading csv/json files. Either "spark" or "pandas".
        spark backend would call spark.read and pandas backend would call pandas.read.
        Default to be "spark".
        """
        return cls._orca_pandas_read_backend

    @orca_pandas_read_backend.setter
    def orca_pandas_read_backend(cls, value):
        value = value.lower()
        assert value == "spark" or value == "pandas", \
            "orca_pandas_read_backend must be either spark or pandas"
        cls._orca_pandas_read_backend = value


class OrcaContext(metaclass=OrcaContextMeta):
    pass
