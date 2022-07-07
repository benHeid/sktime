from pywatts.core.pipeline import Pipeline
from pywatts.modules import FunctionModule
import pandas as pd
import xarray as xr
from sktime.forecasting.pywatts import pyWATTSWrapper

pipeline = Pipeline()

FunctionModule(lambda  x: x +1)(x=pipeline["foo"])

time = pd.date_range('2000-01-01', freq='24H', periods=7)
ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

pyw = pyWATTSWrapper(pipeline)
pyw.fit(ds.to_pandas())
result = pyw.transform(ds.to_pandas())

print(result)