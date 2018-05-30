
# AutoGrad

`autograd` provides automatic differentiation for math operations, so that you can easily build your own *custom loss and layer* (in both Python and Scala), as illustracted below. (See more examples [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/autograd)). Conceptually we use reverse mode together with the chain rule for automatic differentiation. `Variable` is used to record the linkage of the operation history, which would generated a directed acyclic graph for the backward execution. Within the execution graph, leaves are the input `variables` and roots are the output `variables`.

1. Define custom functions using `autograd`
   ```
   from zoo.pipeline.api.autograd import *
   
   def mean_absolute_error(y_true, y_pred):
       return mean(abs(y_true - y_pred), axis=1)
   
   def add_one_func(x):
       return x + 1.0
   ```

2. Define model using Keras-style API and *custom `Lambda` layer*
   ```
   from zoo.pipeline.api.keras.layers import *
   from zoo.pipeline.api.keras.models import *
   model = Sequential().add(Dense(1, input_shape=(2,))) \
                       .add(Lambda(function=add_one_func))
   ```

3. Train model with *custom loss function*
   ```
   model.compile(optimizer = SGD(), loss = mean_absolute_error)
   model.fit(x = ..., y = ...)
   ```