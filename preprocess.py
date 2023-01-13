import pandas as pd
import os

from util import get_features
from util import get_aqi

# -----------------------------------------------------------------------

pr_dir = 'output3'
if not os.path.exists(pr_dir):
    os.makedirs(pr_dir)

Locations = pd.read_csv('city_toposheet.csv')

Features = []
Targets = []
for i, location in Locations.iterrows():
    print(location['city'])
    ft = get_features(location)
    if ft:
        Features.append(ft)
        aqi, aqi_cat = get_aqi(location)
        Targets.append({'location': ft['location'], 'AQI': aqi, 'AQI Category': aqi_cat})
        print(aqi_cat)

pd.DataFrame(Features).to_csv(os.path.join(pr_dir, 'features.csv'), index=False)
pd.DataFrame(Targets).to_csv(os.path.join(pr_dir, 'targets.csv'), index=False)


# -------------------------------------------------------------------
