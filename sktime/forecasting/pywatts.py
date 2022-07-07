import pandas as pd
import xarray as xr

from sktime.base import BaseEstimator
from sktime.forecasting.base import BaseForecaster
from sktime.transformations.base import BaseTransformer


class pyWATTSWrapper(BaseTransformer):

    def __init__(self, pipeline):
        self.pipeline = pipeline
        super().__init__()

    def _transform(self, X, y=None):

        assert isinstance(X, pd.DataFrame)
        result, _  = self.pipeline.test(X.to_xarray())
        return xr.Dataset(result).to_pandas()

    # todo: implement this, mandatory
    def fit(self,X=None, y=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        assert isinstance(X, pd.DataFrame)
        # TODO how to handle X and y here? How to select column here?
        self.pipeline.train(X.to_xarray())
        self._is_fitted = True
