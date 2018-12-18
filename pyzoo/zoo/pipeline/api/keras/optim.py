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
from bigdl.util.common import *
from bigdl.optim.optimizer import OptimMethod, Default

if sys.version >= '3':
    long = int
    unicode = str


class AdamWithSchedule(OptimMethod):
    """

    """
    def __init__(self,
                 learningrate=1e-3,
                 learningrate_decay=0.0,
                 leaningrate_schedule=None,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 epsilon = 1e-8,
                 bigdl_type="float"):
        super(AdamWithSchedule, self).__init__(None, bigdl_type, learningrate, learningrate_decay,
            leaningrate_schedule if (leaningrate_schedule) else Default(), beta1, beta2, epsilon)
