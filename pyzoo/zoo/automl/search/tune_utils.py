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

# Copyright 2017 The Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/tune/suggest/__init__.py
# https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/__init__.py


from ray.tune.suggest.basic_variant import BasicVariantGenerator

from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from ray.tune.schedulers.hyperband import HyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
from ray.tune.schedulers.median_stopping_rule import MedianStoppingRule
from ray.tune.schedulers.pbt import PopulationBasedTraining


# Shim Instantiation is supported after ray 1.0.0. Therefore create_searcher and create_scheduler
# should be removed after ray 1.0.0 is supported.
def create_searcher(
        search_alg,
        **kwargs,
):
    """Instantiate a search algorithm based on the given string.

    This is useful for swapping between different search algorithms.

    Args:
        search_alg (str): The search algorithm to use.
        metric (str): The training result objective value attribute. Stopping
            procedures will use this attribute.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        **kwargs: Additional parameters.
            These keyword arguments will be passed to the initialization
            function of the chosen class.
    Returns:
        ray.tune.suggest.Searcher: The search algorithm.
    """

    def _import_variant_generator():
        return BasicVariantGenerator

    def _import_ax_search():
        from ray.tune.suggest.ax import AxSearch
        return AxSearch

    def _import_dragonfly_search():
        from ray.tune.suggest.dragonfly import DragonflySearch
        return DragonflySearch

    def _import_skopt_search():
        from ray.tune.suggest.skopt import SkOptSearch
        return SkOptSearch

    def _import_hyperopt_search():
        from ray.tune.suggest.hyperopt import HyperOptSearch
        return HyperOptSearch

    def _import_bayesopt_search():
        from ray.tune.suggest.bayesopt import BayesOptSearch
        return BayesOptSearch

    def _import_bohb_search():
        from ray.tune.suggest.bohb import TuneBOHB
        return TuneBOHB

    def _import_nevergrad_search():
        from ray.tune.suggest.nevergrad import NevergradSearch
        return NevergradSearch

    def _import_zoopt_search():
        from ray.tune.suggest.zoopt import ZOOptSearch
        return ZOOptSearch

    def _import_sigopt_search():
        from ray.tune.suggest.sigopt import SigOptSearch
        return SigOptSearch

    SEARCH_ALG_IMPORT = {
        "variant_generator": _import_variant_generator,
        "random": _import_variant_generator,
        "ax": _import_ax_search,
        "dragonfly": _import_dragonfly_search,
        "skopt": _import_skopt_search,
        "hyperopt": _import_hyperopt_search,
        "bayesopt": _import_bayesopt_search,
        "bohb": _import_bohb_search,
        "nevergrad": _import_nevergrad_search,
        "zoopt": _import_zoopt_search,
        "sigopt": _import_sigopt_search,
    }
    search_alg = search_alg.lower()
    if search_alg not in SEARCH_ALG_IMPORT:
        raise ValueError(
            f"Search alg must be one of {list(SEARCH_ALG_IMPORT)}. "
            f"Got: {search_alg}")

    SearcherClass = SEARCH_ALG_IMPORT[search_alg]()
    return SearcherClass(**kwargs)


def create_scheduler(
        scheduler,
        **kwargs,
):
    """Instantiate a scheduler based on the given string.

    This is useful for swapping between different schedulers.

    Args:
        scheduler (str): The scheduler to use.
        **kwargs: Scheduler parameters.
            These keyword arguments will be passed to the initialization
            function of the chosen scheduler.
    Returns:
        ray.tune.schedulers.trial_scheduler.TrialScheduler: The scheduler.
    """

    SCHEDULER_IMPORT = {
        "fifo": FIFOScheduler,
        "async_hyperband": AsyncHyperBandScheduler,
        "asynchyperband": AsyncHyperBandScheduler,
        "median_stopping_rule": MedianStoppingRule,
        "medianstopping": MedianStoppingRule,
        "hyperband": HyperBandScheduler,
        "hb_bohb": HyperBandForBOHB,
        "pbt": PopulationBasedTraining,
    }
    scheduler = scheduler.lower()
    if scheduler not in SCHEDULER_IMPORT:
        raise ValueError(
            f"scheduler must be one of {list(SCHEDULER_IMPORT)}. "
            f"Got: {scheduler}")

    SchedulerClass = SCHEDULER_IMPORT[scheduler]
    return SchedulerClass(**kwargs)
