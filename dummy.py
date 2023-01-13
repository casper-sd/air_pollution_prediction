import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import CountVectorizer

# if not os.path.exists('liss3_data'):
#     os.makedirs('liss3_data')
#
# D = pd.read_csv('sta_coord_2.csv')
# for i, row in D.iterrows():
#     print('------------------------------------------------------------')
#     if not os.path.exists(os.path.join('liss3_data', row['state_ut'])):
#         os.makedirs(os.path.join('liss3_data', row['state_ut']))
#     print('Download {city}, {state_ut}'.format(city=row['city'], state_ut=row['state_ut']))
#     pth = os.path.join('liss3_data', row['state_ut'], row['city'])
#     if os.path.exists(pth):
#         print(f'Already {len(os.listdir(pth))} year data available. Download more if required')
#         input()
#     else:
#         os.makedirs(pth)
#         input()
#         print(f'Downloaded {len(os.listdir(pth))} year data.')
#
#

# from functools import reduce
#

# c = plt.imread('liss3_data/Arunachal Pradesh/Itanagar/2015/BAND5.tif')
# plt.imshow(c)
# plt.show()

from itertools import permutations

# D = pd.read_csv('processed_data/dataset.csv')
# D['AQI Category'] = np.nan
# for i, row in D.iterrows():
#     if row['AQI'] <= 50:
#         D.loc[i, 'AQI Category'] = 'GOOD'
#     elif 50 < row['AQI'] <= 100:
#         D.loc[i, 'AQI Category'] = 'MODERATE'
#     elif 100 < row['AQI'] <= 200:
#         D.loc[i, 'AQI Category'] = 'UNHEALTHY'
#     elif 200 < row['AQI'] <= 300:
#         D.loc[i, 'AQI Category'] = 'VERY UNHEALTHY'
#     else:
#         D.loc[i, 'AQI Category'] = 'HAZARDOUS'
#
# D.to_csv('AQI_Category.csv', index=False)

# from functools import reduce
# fns = ['so2', 'no2', 'pm10', 'pm25']
# f = [pd.read_csv('cpcb_data/' + fn + '.csv') for fn in fns]
# pf = []
# for ff in f:
#     ff = ff.groupby(by=['state', 'city']).mean().round(3)
#     ff.reset_index(inplace=True)
#     pf.append(ff)
#
# d = reduce(lambda left, right: pd.merge(left, right, on=['state', 'city'], how='outer'), pf)
# d.to_csv('air_data.csv', index=False)

D = pd.read_csv('output2/targets.csv')
D['Custom AQI Category'] = np.nan

lim = np.linspace(D['AQI'].min(), D['AQI'].max() + 1, 4)
for i, row in D.iterrows():
    if lim[0] <= row['AQI'] < lim[1]:
        D.loc[i, 'Custom AQI Category'] = 'GOOD'
    elif lim[1] <= row['AQI'] < lim[2]:
        D.loc[i, 'Custom AQI Category'] = 'MODERATE'
    elif lim[2] <= row['AQI'] < lim[3]:
        D.loc[i, 'Custom AQI Category'] = 'UNHEALTHY'

D.to_csv('output2/targets.csv', index=False)