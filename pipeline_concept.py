import inspect
from copy import deepcopy
from inspect import _ParameterKind

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.ets import AutoETS

from sktime.forecasting.arima import AutoARIMA

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster

from sktime.transformations.series.exponent import ExponentTransformer

from sktime.base import BaseEstimator

class Pipeline(BaseEstimator):

    def __init__(self, steps):
        super().__init__()
        self.steps =  steps

    @staticmethod
    def _check_validity(step, method_name, **kwargs):
        use_kwargs = {}
        if not hasattr(step, method_name):
            raise Exception(f"Method {method_name} does not exist for {step.__name__}")
        method = getattr(step, method_name)
        method_signature = inspect.signature(method).parameters

        for name, param in method_signature.items():
            if name == "self":
                continue
            if name not in kwargs and param.default is inspect._empty and param.kind != _ParameterKind.VAR_KEYWORD:
                raise Exception(f"Necesssary parameter {name} of method {method_name} is not provided")
            if name in kwargs:
                use_kwargs[name] = kwargs[name]
        return use_kwargs

    def fit(self, **kwargs):
        kwargs = deepcopy(kwargs)
        for transformer in self.steps[:-1]:
            required_kwargs = self._check_validity(transformer, "fit_transform", **kwargs)
            X = transformer.fit_transform(**required_kwargs)
            kwargs["X"] = X
        # fit forecaster
        required_kwargs = self._check_validity(self.steps[-1], "_fit", **kwargs)
        f = self.steps[-1]
        f.fit(**required_kwargs)
        return self

    def transform(self, *args, **kwargs):
        required_kwargs = self._check_validity(self.steps[-1], "transform", **kwargs)
        # TODO additional stuff
    def predict(self, *args, **kwargs):
        kwargs = deepcopy(kwargs)
        for transformer in self.steps[:-1]:
            required_kwargs = self._check_validity(transformer, "transform", **kwargs)
            X = transformer.transform(**required_kwargs)
            kwargs["X"] = X
        required_kwargs = self._check_validity(self.steps[-1], "_predict", **kwargs)
        f = self.steps[-1]
        return f.predict(**required_kwargs)

    def predict_quantiles(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    # Create some data
    data = pd.DataFrame(np.random.normal(0, 1, 100))
    X = pd.DataFrame(data * 2)

    train_y = data[:60]
    train_X = X[:60]
    test_X = X[60:]

    # Build pipeline with forecaster
    pipe = Pipeline([ExponentTransformer(), AutoARIMA()])
    # Fit pipeline
    pipe.fit(y=train_y, X=train_X) # TODO Only kwargs are possible
    # Predict pipeline.
    result = pipe.predict(fh=list(range(1, 41)), X=test_X)

    plt.plot(result)
    plt.show()