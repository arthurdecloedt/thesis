import numpy as np
import pandas as pd

momentsel = 4

moments = pd.read_csv('resources/vixcurrent.csv', index_col=0, usecols=[0, momentsel], parse_dates=True).dropna()

dates = moments.index.to_pydatetime()
dates_np = np.array([np.datetime64(d.date()) for d in dates])
mom_array = moments[moments.columns[0]].to_numpy()

vixd = pd.read_csv('resources/vixaapl.csv', index_col=0, usecols=[0, momentsel], parse_dates=True).dropna()

dates = vixd.index.to_pydatetime()
vdates_np = np.array([np.datetime64(d.date()) for d in dates])
vix_array = vixd[vixd.columns[0]].to_numpy()

_, dind, vind = np.intersect1d(dates_np, vdates_np, return_indices=True)

mom_array = mom_array[dind]
vix_array = vix_array[vind]

arr = np.array([mom_array, vix_array])
coef = np.corrcoef(arr)

autocor = np.corrcoef(np.array([vind[:1], vind[1:]))
