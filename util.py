import numpy as np
import matplotlib.pyplot as plt
import os
from xml.etree.ElementTree import parse
from itertools import product
import pandas as pd
import warnings


l3_root_dir = 'liss3_data'
L3_SPACIAL_RES = 24  # In KM

threshold_range = [-1, 0, 1]
th_range = list(product(threshold_range, repeat=4))
# th_range = list(filter(lambda p: -1 <= sum(p) <= 1, perm_th_range))
th_range.remove((0, 0, 0, 0))
FILTERS = {f'filter_{i}': flt for i, flt in enumerate(th_range)}
print(f'Number of filters: {len(th_range)}')

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

actual_bbox_df = pd.read_csv('city_bounds.csv')
actual_bbox_df.set_index(['state', 'city'], inplace=True)


# ----------------------------------------------------------------------------
def get_tile_bbox(tile: str):
    info = parse(os.path.join(l3_root_dir, tile, tile+'_info.xml'))
    min_str = info.find('./Coverage/Lower_left').text.strip('\n').replace(' ', '').split(',')
    max_str = info.find('./Coverage/Upper_right').text.strip('\n').replace(' ', '').split(',')
    return {
        'tile': tile,
        'min_lat': float(min_str[1][2:-1]),
        'min_lon': float(min_str[0][2:-1]),
        'max_lat': float(max_str[1][2:-1]),
        'max_lon': float(max_str[0][2:-1])
    }


def get_merged_tile(tiles: list, band_name: str):
    s_tiles = [get_tile_bbox(tile) for tile in tiles]
    s_tiles.sort(key=lambda x: x['min_lon'])
    s_tiles.sort(key=lambda x: x['min_lat'], reverse=True)
    tiles_data = [plt.imread(os.path.join(l3_root_dir,
                                          tile['tile'],
                                          tile['tile']+'_'+band_name+'.tif'))
                  for tile in s_tiles]

    constraints = {
        'min_lat': min([s_tiles[i]['min_lat'] for i in range(len(tiles))]),
        'max_lat': max([s_tiles[i]['max_lat'] for i in range(len(tiles))]),
        'min_lon': min([s_tiles[i]['min_lon'] for i in range(len(tiles))]),
        'max_lon': max([s_tiles[i]['max_lon'] for i in range(len(tiles))])
    }

    if len(tiles) == 4:
        s1 = np.concatenate((tiles_data[0][:, :1131], tiles_data[1][:, 20:]), axis=1)
        s2 = np.concatenate((tiles_data[2][:, :1131], tiles_data[3][:, 20:]), axis=1)
        return np.concatenate((s1[:1131, :], s2[20:, :]), axis=0), constraints
    elif len(tiles) == 2:
        if s_tiles[0]['min_lat'] == s_tiles[1]['min_lat']:
            return np.concatenate((tiles_data[0][:, :1131], tiles_data[1][:, 20:]), axis=1), constraints
        else:
            return np.concatenate((tiles_data[0][:1131, :], tiles_data[1][20:, :]), axis=0), constraints
    elif len(tiles) == 1:
        return tiles_data[0], constraints
    else:
        raise Exception(f'Incorrect numbers of tiles: {len(tiles)}')


def get_cropped_tile(a_cons, d_cons, shape):
    left_ind, right_ind, top_ind, bottom_ind = 0, shape[1], 0, shape[0]
    d_lat = d_cons['max_lat'] - d_cons['min_lat']
    d_lon = d_cons['max_lon'] - d_cons['min_lon']
    if d_cons['max_lon'] > a_cons['min_lon'] > d_cons['min_lon']:
        left_ind = int(shape[1] * (a_cons['min_lon'] - d_cons['min_lon']) / d_lon)
    if d_cons['min_lon'] < a_cons['max_lon'] < d_cons['max_lon']:
        right_ind = int(shape[1] * (a_cons['max_lon'] - d_cons['min_lon']) / d_lon)
    if d_cons['max_lat'] > a_cons['min_lat'] > d_cons['min_lat']:
        bottom_ind = int(shape[0] - shape[0] * (a_cons['min_lat'] - d_cons['min_lat']) / d_lat)
    if d_cons['min_lat'] < a_cons['max_lat'] < d_cons['max_lat']:
        top_ind = int(shape[0] - shape[0] * (a_cons['max_lat'] - d_cons['min_lat']) / d_lat)

    return left_ind, right_ind, top_ind, bottom_ind


def get_features(loc_data):
    features = {'location': loc_data['state'] + '-' + loc_data['city']}
    tile_names = []
    for t in range(1, 5):
        tile = loc_data[f'Toposheet_{t}']
        if not pd.isna(tile):
            tile_names.append(tile)

    B2, B2_cons = get_merged_tile(tile_names, 'BAND2')
    B3, B3_cons = get_merged_tile(tile_names, 'BAND3')
    B4, B4_cons = get_merged_tile(tile_names, 'BAND4')
    B5, B5_cons = get_merged_tile(tile_names, 'BAND5')

    a_cons = actual_bbox_df.loc[loc_data['state'], loc_data['city']]
    left_ind, right_ind, top_ind, bottom_ind = get_cropped_tile(a_cons, B2_cons, B2.shape)
    B2C = B2[top_ind:bottom_ind, left_ind:right_ind]
    B3C = B3[top_ind:bottom_ind, left_ind:right_ind]
    B4C = B4[top_ind:bottom_ind, left_ind:right_ind]
    B5C = B5[top_ind:bottom_ind, left_ind:right_ind]

    plt.imshow(B2C, cmap='gray')
    plt.show()

    n_pixel = (right_ind - left_ind) * (bottom_ind - top_ind)
    if n_pixel == 0:
        warnings.warn('Invalid Cropping: {city}'.format(city=features['location']))
        return None

    for f_name in FILTERS.keys():
        flt = FILTERS[f_name]
        B = flt[0] * B2C + flt[1] * B3C + flt[2] * B4C + flt[3] * B5C
        # m = flt[0] * np.mean(B2C) + flt[1] * np.mean(B3C) + flt[2] * np.mean(B4C) + flt[3] * np.mean(B5C)

        th = np.mean(flt)
        nB = B / np.amax(np.abs(B))
        features[f_name] = np.sum(nB > th) / n_pixel
        BeX = np.where(nB > th, nB, 0)
        plt.title(flt)
        plt.imshow(BeX)
        plt.show()

    return features


def calc_subindex(param, val):
    b_path = os.path.join('aqi_breakpoints', param + '_aqi_breakpoints.csv')
    bp_df = pd.read_csv(b_path)
    for i, rr in bp_df.iterrows():
        if rr['Low Breakpoint'] <= val < rr['High Breakpoint']:
            c_ratio = (val - rr['Low Breakpoint']) / (rr['High Breakpoint'] - rr['Low Breakpoint'])
            return rr['Low AQI'] + (rr['High AQI'] - rr['Low AQI']) * c_ratio, rr['AQI Category']


def get_aqi(loc_data):
    sub_indices = []
    for pollutant in ['NO2', 'SO2', 'PM10', 'PM2.5']:
        v = loc_data[pollutant+'_AVG']
        if not pd.isna(v):
            a = calc_subindex(pollutant, v)
            if a:
                sub_indices.append(a)
    if len(sub_indices) < 3:
        return np.nan, np.nan
    else:
        return max(sub_indices, key=lambda x: x[0])


D = pd.read_csv('city_toposheet.csv')
get_features(D.iloc[196])