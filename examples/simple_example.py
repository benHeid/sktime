from pywatts.core.pipeline import Pipeline
from pywatts.modules import SKLearnWrapper, FunctionModule
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.arima import ARIMA



from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_unit_test



from sktime.forecasting.compose import MultioutputTimeSeriesRegressionForecaster,\
MultioutputTabularRegressionForecaster,\
RecursiveTimeSeriesRegressionForecaster, \
RecursiveTabularRegressionForecaster
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split



# %% ARIMA example
print('\nARIMA example')
y = load_airline()
print(y.tail())
y, X = load_longley()


X["target"] = y
import pandas as pd
X.index = pd.date_range(pd.Timestamp(year=1947, month=1, day=1), periods=len(X), freq="1y")

forecaster = ARIMA()

pipeline = Pipeline()

scaler = SKLearnWrapper(StandardScaler())(y=pipeline["target"])
scaler = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape(-1), x, "reshaped"))(x=scaler)
arima_forecast = ARIMA()(y=scaler, fh=[1, 2, 3])

pipeline.train(X)




forecaster.fit(y) # fh not required



print('\nARIMA out-of-sample')
y_pred = forecaster.predict(fh=[1, 2, 3])
print(y_pred)



print('\nARIMA in-sample')
y_pred = forecaster.predict(fh=[-2, -1, 0])
print(y_pred)



print('\nARIMA in-sample and out-of-sample')
y_pred = forecaster.predict(fh=[-2, -1, 0, 1, 2, 3])
print(y_pred)



# fh class used to differentiate between in-sample and out-of-sample



# %% kNN classifier example



print('\nkNN classifier example')
X_train, y_train = load_unit_test(return_X_y=True, split="train")
print(X_train.tail())



X_test, y_test = load_unit_test(return_X_y=True, split="test")



classifier = KNeighborsTimeSeriesClassifier()



classifier.fit(X_train, y_train)
KNeighborsTimeSeriesClassifier(...)



print('\nkNN classifier out-of-sample')
y_pred = classifier.predict(X_test)
print(y_pred)



print('\nkNN classifier in-sample')
y_pred = classifier.predict(X_train)
print(y_pred)



# possible problem for pyWATTS
# inputs of fit and predict between classes BaseForecaster and BaseClassifier differ



# %% Linear Regression Multiple-Output strategy example'



print('\nLinear Regression Multiple-Output strategy example')
y, X = load_longley()
print(y.tail())
print(X.tail())



y_train, _, X_train, X_test = temporal_train_test_split(y, X)



forecaster = MultioutputTabularRegressionForecaster(estimator=LinearRegression(), window_length=3)
# forecaster = MultioutputTimeSeriesRegressionForecaster(estimator=LinearRegression(), window_length=3)



fh_is = ForecastingHorizon(X_train.index, is_relative=False)
fh_oos = ForecastingHorizon(X_test.index, is_relative=False)



forecaster.fit(y=y_train, X=X_train, fh=fh_oos) # fh required for fit



print('\nLinear Regression Multiple-Output strategy out-of-sample')
y_pred = forecaster.predict(X=X)
# The forecasting horizon `fh` must be passed either to `fit` or `predict`.
print(y_pred)



# print('\nLinear Regression Multiple-Output strategy in-sample')
# # forecaster needs to be refitted with fh_is
# # in-sample predictions not implemented yet
# forecaster.fit(y=y_train, X=X_train, fh=fh_is) # fh required
# y_pred = forecaster.predict(X=X, fh=fh_is)
# print(y_pred)



# %% Linear Regression Recursive strategy example



print('\nLinear Regression Recursive strategy example')
y, X = load_longley()
print(y.tail())
print(X.tail())



y_train, _, X_train, X_test = temporal_train_test_split(y, X)



# what is the difference between Time Series Regression and Tabular Regression? Time Series Regression does not work.
# forecaster = RecursiveTimeSeriesRegressionForecaster(estimator=LinearRegression(), window_length=10)
forecaster = RecursiveTabularRegressionForecaster(estimator=LinearRegression(), window_length=10)
# window_length has no influence



fh_is = ForecastingHorizon(X_train.index, is_relative=False)
fh_oos = ForecastingHorizon(X_test.index, is_relative=False)



forecaster.fit(y=y_train, X=X_train) # fh not required for fit



# print('\nLinear Regression Multiple-Output strategy in-sample')
# # in-sample predictions not implemented yet
# y_pred = forecaster.predict(X=X_train, fh=fh_is)
# print(y_pred)



print('\nLinear Regression Multiple-Output strategy out-of-sample')
y_pred = forecaster.predict(X=X_test, fh=fh_oos)
# The forecasting horizon `fh` must be passed either to `fit` or `predict`.
print(y_pred)