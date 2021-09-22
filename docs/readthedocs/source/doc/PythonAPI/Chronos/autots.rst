AutoTS (deprecated)
=====================

.. warning::
    The API in this page will be deprecated soon.

AutoTSTrainer
----------------------------------------

AutoTSTrainer trains a time series pipeline (including data processing, feature engineering, and model) with AutoML.

.. autoclass:: zoo.chronos.autots.deprecated.forecast.AutoTSTrainer
    :members:
    :show-inheritance:


TSPipeline
----------------------------------------

A pipeline for time series forecasting.

.. autoclass:: zoo.chronos.autots.deprecated.forecast.TSPipeline
    :members:
    :show-inheritance:


chronos.config.recipe
----------------------------------------

Recipe is used for search configuration for AutoTSTrainer.

.. autoclass:: zoo.chronos.autots.deprecated.config.recipe.SmokeRecipe
    :members:
    :show-inheritance:

.. autoclass:: zoo.chronos.autots.deprecated.config.recipe.MTNetSmokeRecipe
    :members:
    :show-inheritance:

.. autoclass:: zoo.chronos.autots.deprecated.config.recipe.TCNSmokeRecipe
    :members:
    :show-inheritance:

.. autoclass:: zoo.chronos.autots.deprecated.config.recipe.PastSeqParamHandler
    :members:
    :show-inheritance: