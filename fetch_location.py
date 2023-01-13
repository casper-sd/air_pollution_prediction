import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd

geolocator = Nominatim(user_agent="my-app")

sta = pd.read_csv('locations.csv')[['state_ut', 'place']].copy()

sta['lat'] = np.nan
sta['lon'] = np.nan
sta['n_stations'] = np.nan

for i, row in sta.iterrows():
    print(row['place'] + ', ' + row['state_ut'])
    location = geolocator.geocode(row['place'] + ', ' + row['state_ut'], country_codes='IN')
    sta.loc[i, 'lat'] = location.latitude
    sta.loc[i, 'lon'] = location.longitude

sta.to_csv('locations.csv', index=False)

# sta = pd.read_csv('locations.csv')
# ST = pd.read_csv('Stations_NAMP.csv', dtype=str)
# sm = pd.merge(sta, ST, on=['state_ut', 'city'], how='left')
# sm.to_csv('stations_location.csv', index=False)

