import pandas as pd
from src.rebase_dplyr_to_pandas import arrange, distinct, filter, pull, count, add_count, mutate, rename, relocate
import numpy as np
import unittest
import pytest


class TestDplyrToPandas(unittest.TestCase):

    # Arrange
    def test_arrange_pandas(self):
        cars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\mtcars.csv')
        cars = cars.rename({'Unnamed: 0': 'car'}, axis=1)
        expected = pd.DataFrame({'car': ['Toyota Corolla', 'Honda Civic', 'Fiat 128', 'Fiat X1-9', 'Lotus Europa'],
                                 'mpg': [33.9, 30.4, 32.4, 27.3, 30.4], 'cyl': [4] * 5,
                                 'disp': [71.1, 75.7, 78.7, 79.0, 95.1],
                                 'hp': [65, 52, 66, 66, 113],
                                 'drat': [4.22, 4.93, 4.08, 4.08, 3.77],
                                 'wt': [1.835, 1.615, 2.200, 1.935, 1.513],
                                 'qsec': [19.90, 18.52, 19.47, 18.90, 16.90],
                                 'vs': [1] * 5, 'am': [1] * 5, 'gear': [4] * 4 + [5],
                                 'carb': [1, 2, 1, 1, 2]})
        actual = arrange(cars, ['cyl', 'disp']).head()
        pd.testing.assert_frame_equal(expected, actual)

    def test_arrange_pandasDesc(self):
        cars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\mtcars.csv')
        cars = cars.rename({'Unnamed: 0': 'car'}, axis=1)
        actual = arrange(cars, "desc(disp)").head()
        expected = pd.DataFrame({'car': ['Cadillac Fleetwood', 'Lincoln Continental', 'Chrysler Imperial', 'Pontiac Firebird', 'Hornet Sportabout'],
                                 'mpg': [10.4, 10.4, 14.7, 19.2, 18.7], 'cyl': [8] * 5,
                                 'disp': [472.0, 460.0, 440.0, 400.0, 360.0],
                                 'hp': [205, 215, 230, 175, 175],
                                 'drat': [2.93, 3.00, 3.23, 3.08, 3.15],
                                 'wt': [5.250, 5.424, 5.345, 3.845, 3.440],
                                 'qsec': [17.98, 17.82, 17.42, 17.05, 17.02],
                                 'vs': [0] * 5, 'am': [0] * 5, 'gear': [3] * 5, 'carb': [4, 4, 4, 2, 2]})
        pd.testing.assert_frame_equal(expected, actual)

    # Count
    def test_count_pandas(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        expected = pd.DataFrame({'sex': ['male', 'female', 'none', 'none', 'hermaphroditic'],
                                 'gender': ['masculine', 'feminine', 'masculine', 'feminine', 'masculine'],
                                 'n': [60., 16., 5., 1., 1.]})
        expected['sex'] = expected['sex'].astype('category')
        expected['gender'] = expected['gender'].astype('category')
        pd.testing.assert_frame_equal(count(starwars, ['sex', 'gender'], sort=True), expected)

    def test_count_pandasWeight(self):
        df = pd.DataFrame({'name': ['Max', 'Sandra', 'Susan'],
                           'gender': ['male', 'female', 'female'],
                           'runs': [10, 1, 4]})
        expected = pd.DataFrame({'gender': ['female', 'male'],
                                 'n': [5, 10]})
        expected['gender'] = expected['gender'].astype('category')
        pd.testing.assert_frame_equal(count(df, 'gender', wt='runs'), expected)

    # Add Count
    def test_addCount_pandas(self):
        df = pd.DataFrame({'name': ['Max', 'Sandra', 'Susan'],
                           'gender': ['male', 'female', 'female'],
                           'runs': [10, 1, 4]})
        expected = pd.DataFrame({'name': ['Max', 'Sandra', 'Susan'],
                                 'gender': ['male', 'female', 'female'],
                                 'runs': [10, 1, 4],
                                 'n': [10, 5, 5]})
        pd.testing.assert_frame_equal(add_count(df, "gender", wt="runs"), expected)

    # Distinct
    def test_distinct_pandasAll(self):
        data = pd.DataFrame({'x': np.random.choice(10, 100, replace=True), 'y': np.random.choice(10, 100, replace=True)})
        assert len(distinct(data)) == len(distinct(data, ['x', 'y']))

    def test_distinct_pandasColumn(self):
        data = pd.DataFrame({'x': np.random.choice(10, 100, replace=True), 'y': np.random.choice(10, 100, replace=True)})
        assert len(distinct(data, 'x')) == 10

    def test_distinct_pandasKeepAll(self):
        data = pd.DataFrame({'x': np.random.choice(10, 100, replace=True), 'y': np.random.choice(10, 100, replace=True)})
        assert distinct(data, 'y', keep_all=True).shape == (10, 2)

    # Filter
    def test_filter_pandas(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        expected = pd.DataFrame({'name': ['Jabba Desilijic Tiure'], 'height': [175.0], 'mass': [1358.0],
                                 'hair_color': [np.nan], 'skin_color': ['green-tan, brown'], 'eye_color': ['orange'],
                                 'birth_year': [600.0], 'sex': ['hermaphroditic'], 'gender': ['masculine'],
                                 'homeworld': ['Nal Hutta'], 'species': ['Hutt']})
        expected['hair_color'] = expected['hair_color'].astype('object')
        pd.testing.assert_frame_equal(expected, filter(starwars, 'mass > 1000'))

    def test_filter_pandasMean(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        expected = pd.DataFrame({'name': ['Darth Vader', 'Owen Lars', 'Chewbacca', 'Jabba Desilijic Tiure', 'Jek Tono Porkins'],
                                 'height': [202.0, 178.0, 228.0, 175.0, 180.0], 'mass': [136.0, 120.0, 112.0, 1358.0, 110.0],
                                 'hair_color': ['none', 'brown, grey', 'brown', np.nan, 'brown'],
                                 'skin_color': ['white', 'light', 'unknown', 'green-tan, brown', 'fair'],
                                 'eye_color': ['yellow', 'blue', 'blue', 'orange', 'blue'],
                                 'birth_year': [41.9, 52.0, 200.0, 600.0, np.nan], 'sex': ['male', 'male', 'male', 'hermaphroditic', 'male'],
                                 'gender': ['masculine'] * 5, 'homeworld': ['Tatooine', 'Tatooine', 'Kashyyyk', 'Nal Hutta', 'Bestine IV'],
                                 'species': ['Human', 'Human', 'Wookiee', 'Hutt', 'Human']})
        pd.testing.assert_frame_equal(expected, filter(starwars, 'mass > mean(mass)').head())

    def test_filter_pandasMedian(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        expected = pd.DataFrame(
            {'name': ['Darth Vader', 'Owen Lars', 'Biggs Darklighter', 'Anakin Skywalker', 'Chewbacca'],
             'height': [202.0, 178.0, 183.0, 188.0, 228.0], 'mass': [136.0, 120.0, 84.0, 84.0, 112.0],
             'hair_color': ['none', 'brown, grey', 'black', 'blond', 'brown'],
             'skin_color': ['white', 'light', 'light', 'fair', 'unknown'],
             'eye_color': ['yellow', 'blue', 'brown', 'blue', 'blue'],
             'birth_year': [41.9, 52.0, 24.0, 41.9, 200.0], 'sex': ['male'] * 5,
             'gender': ['masculine'] * 5, 'homeworld': ['Tatooine', 'Tatooine', 'Tatooine', 'Tatooine', 'Kashyyyk'],
             'species': ['Human', 'Human', 'Human', 'Human', 'Wookiee']})
        pd.testing.assert_frame_equal(expected, filter(starwars, "mass > median(mass)").head())

    def test_filter_pandasMax(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        expected = pd.DataFrame({'name': ['Jabba Desilijic Tiure'], 'height': [175.0], 'mass': [1358.0],
                                 'hair_color': [np.nan], 'skin_color': ['green-tan, brown'], 'eye_color': ['orange'],
                                 'birth_year': [600.0], 'sex': ['hermaphroditic'], 'gender': ['masculine'],
                                 'homeworld': ['Nal Hutta'], 'species': ['Hutt']})
        expected['hair_color'] = expected['hair_color'].astype('object')
        pd.testing.assert_frame_equal(expected, filter(starwars, 'mass == max(mass)'))

    def test_filter_pandasMin(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        expected = pd.DataFrame({'name': ['Ratts Tyerell'], 'height': [79.0], 'mass': [15.0], 'hair_color': ['none'],
                                 'skin_color': ['grey, blue'], 'eye_color': ['unknown'], 'birth_year': [np.nan],
                                 'sex': ['male'], 'gender': ['masculine'], 'homeworld': ['Aleen Minor'], 'species': ['Aleena']})
        pd.testing.assert_frame_equal(expected, filter(starwars, 'mass == min(mass)'))

    def test_filter_pandasQuantile(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        pd.testing.assert_frame_equal(filter(starwars, "mass==median(mass)"), filter(starwars, "mass==quantile(mass, 0.5)"))

    # Mutate
    def test_mutate_pandasString(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 9, 8],
                                 'z': [11, 11, 11]})
        pd.testing.assert_frame_equal(mutate(data, 'z = x + y'), expected)

    def test_mutate_pandasString_log(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y1': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y1': [10, 9, 8],
                                 'z': [2.302585, 2.197225, 2.079442]})
        pd.testing.assert_frame_equal(mutate(data, 'z = log(y1)'), expected)

    def test_mutate_pandasString_log2(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y_1': [2, 4, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y_1': [2, 4, 8],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, 'z = log2(y_1)'), expected)

    def test_mutate_pandasString_log10(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'yY': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'yY': [10, 100, 1000],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, 'z = log10(yY)'), expected)

    def test_mutate_pandasString_arcsin(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 9, 8],
                                 'z': [1.000000, 1.141593, 0.141593]})
        pd.testing.assert_frame_equal(mutate(data, "z = arcsin(sin(x))"), expected)

    def test_mutate_pandasString_arccos(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 9, 8],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = arccos(cos(x))"), expected)

    def test_mutate_pandasString_arctan(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 9, 8],
                                 'z': [1.000000, -1.141593, -0.141593]})
        pd.testing.assert_frame_equal(mutate(data, "z = arctan(tan(x))"), expected)

    def test_mutate_pandasString_arcsinh(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 9, 8],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = arcsinh(sinh(x))"), expected)

    def test_mutate_pandasString_arccosh(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 9, 8],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = arccosh(cosh(x))"), expected)

    def test_mutate_pandasString_arctanh(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 9, 8],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = arctanh(tanh(x))"), expected)

    def test_mutate_pandasString_ceil(self):
        data = pd.DataFrame({'x': [1.2, 2.5, 3.7],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1.2, 2.5, 3.7],
                                 'y': [10, 9, 8],
                                 'z': [2.0, 3.0, 4.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = ceil(x)"), expected)

    def test_mutate_pandasString_floor(self):
        data = pd.DataFrame({'x': [1.2, 2.5, 3.7],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1.2, 2.5, 3.7],
                                 'y': [10, 9, 8],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = floor(x)"), expected)

    def test_mutate_pandasString_round(self):
        data = pd.DataFrame({'x': [1.2, 2.5, 3.7],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1.2, 2.5, 3.7],
                                 'y': [10, 9, 8],
                                 'z': [1.0, 2.0, 4.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = round(x, 0)"), expected)

    def test_mutate_pandasString_sqrt(self):
        data = pd.DataFrame({'x': [1, 4, 9],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [1, 4, 9],
                                 'y': [10, 9, 8],
                                 'z': [1.0, 2.0, 3.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = sqrt(x)"), expected)

    def test_mutate_pandasString_abs(self):
        data = pd.DataFrame({'x': [-1, 4, -9],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [-1, 4, -9],
                                 'y': [10, 9, 8],
                                 'z': [1, 4, 9]})
        pd.testing.assert_frame_equal(mutate(data, "z = abs(x)"), expected)

    def test_mutate_pandasString_sign(self):
        data = pd.DataFrame({'x': [-1, 4, -9],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [-1, 4, -9],
                                 'y': [10, 9, 8],
                                 'z': [-1, 1, -1]})
        pd.testing.assert_frame_equal(mutate(data, "z = sign(x)"), expected)

    def test_mutate_pandasString_mean(self):
        data = pd.DataFrame({'x': [-1, 4, -9],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [-1, 4, -9],
                                 'y': [10, 9, 8],
                                 'z': [9.0, 9.0, 9.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = mean(y)"), expected)

    def test_mutate_pandasString_median(self):
        data = pd.DataFrame({'x': [-1, 4, -9],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [-1, 4, -9],
                                 'y': [10, 9, 8],
                                 'z': [9.0, 9.0, 9.0]})
        pd.testing.assert_frame_equal(mutate(data, "z = median(y)"), expected)

    def test_mutate_pandasString_min(self):
        data = pd.DataFrame({'x': [-1, 4, -9],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [-1, 4, -9],
                                 'y': [10, 9, 8],
                                 'z': [8, 8, 8]})
        pd.testing.assert_frame_equal(mutate(data, "z = min(y)"), expected)

    def test_mutate_pandasString_max(self):
        data = pd.DataFrame({'x': [-1, 4, -9],
                             'y': [10, 9, 8]})
        expected = pd.DataFrame({'x': [-1, 4, -9],
                                 'y': [10, 9, 8],
                                 'z': [10, 10, 10]})
        pd.testing.assert_frame_equal(mutate(data, "z = max(y)"), expected)

    def test_mutate_pandasString_cumsum(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [1, 3, 6]})
        pd.testing.assert_frame_equal(mutate(data, 'z = cumsum(x)'), expected)

    def test_mutate_pandasString_cummin(self):
        data = pd.DataFrame({'x': [3, 4, 2, 4, 1],
                             'y': [10, 100, 1000, 10000, 100000]})
        expected = pd.DataFrame({'x': [3, 4, 2, 4, 1],
                                 'y': [10, 100, 1000, 10000, 100000],
                                 'z': [3, 3, 2, 2, 1]})
        pd.testing.assert_frame_equal(mutate(data, 'z = cummin(x)'), expected)

    def test_mutate_pandasString_cummax(self):
        data = pd.DataFrame({'x': [3, 4, 2, 5, 1],
                             'y': [10, 100, 1000, 10000, 100000]})
        expected = pd.DataFrame({'x': [3, 4, 2, 5, 1],
                                 'y': [10, 100, 1000, 10000, 100000],
                                 'z': [3, 4, 4, 5, 5]})
        pd.testing.assert_frame_equal(mutate(data, 'z = cummax(x)'), expected)

    def test_mutate_pandasString_lag(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [np.nan, 1.0, 2.0]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lag(x)'), expected)

    def test_mutate_pandasString_lagNEquals(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [np.nan, np.nan, 1.0]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lag(x, n=2)'), expected)

    def test_mutate_pandasString_lagDefaultEquals(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [0, 1, 2]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lag(x, default=0)'), expected)

    def test_mutate_pandasString_lagNoKwargs(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [0, 0, 1]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lag(x, 2, 0)'), expected)

    def test_mutate_pandasString_lead(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [2.0, 3.0, np.nan]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lead(x)'), expected)

    def test_mutate_pandasString_leadNEquals(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [3.00, np.nan, np.nan]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lead(x, n=2)'), expected)

    def test_mutate_pandasString_leadDefaultEquals(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [2, 3, 0]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lead(x, default=0)'), expected)

    def test_mutate_pandasString_leadNoKwargs(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [3, 0, 0]})
        pd.testing.assert_frame_equal(mutate(data, 'z = lead(x, 2, 0)'), expected)

    def test_mutate_pandasString_ifelse(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [True, False, False]})
        pd.testing.assert_frame_equal(mutate(data, 'z = if_else(x == 1, True, False)'), expected)

    def test_mutate_pandasString_naif(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, 1000],
                                 'z': [np.nan, 2, 3]})
        pd.testing.assert_frame_equal(mutate(data, 'z = na_if(x, 1)'), expected)

    def test_mutate_pandasString_coalesce(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, np.nan]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [10, 100, np.nan],
                                 'z': [10., 100., 1000.]})
        pd.testing.assert_frame_equal(mutate(data, 'z = coalesce(y, 1000)'), expected)

    # Pull
    def test_pull_pandas(self):
        cars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\mtcars.csv')
        expected = pd.Series([4, 4, 1, 1, 2], name='carb')
        pd.testing.assert_series_equal(expected, pull(cars, -1).head())

    def test_pull_pandasName(self):
        starwars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\starwars.csv')
        expected = pd.Series([172.0, 167.0, 96.0, 202.0, 150.0], name='height')
        expected.index = pd.Series(['Luke Skywalker', 'C-3PO', 'R2-D2', 'Darth Vader', 'Leia Organa'], name='name')
        pd.testing.assert_series_equal(expected, pull(starwars, 'height', 'name').head())

    # Rename
    def test_rename_pandasString(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'blarg': [1, 2, 3],
                                 'y': [10, 100, 1000]})
        pd.testing.assert_frame_equal(rename(data, 'x = blarg'), expected)

    def test_rename_pandasList(self):
        data = pd.DataFrame({'x': [1, 2, 3],
                             'y': [10, 100, 1000]})
        expected = pd.DataFrame({'blarg': [1, 2, 3],
                                 'smarg': [10, 100, 1000]})
        pd.testing.assert_frame_equal(rename(data, ['x = blarg', 'y = smarg']), expected)

    # Relocate
    def test_relocate_beforeAfterNone(self):
        data = pd.DataFrame({'a': [1],
                             'b': [1],
                             'c': [1],
                             'd': ['a'],
                             'e': ['a'],
                             'f': ['a']})
        expected = pd.DataFrame({'f': ['a'],
                                 'a': [1],
                                 'b': [1],
                                 'c': [1],
                                 'd': ['a'],
                                 'e': ['a']})
        pd.testing.assert_frame_equal(relocate(data, 'f'), expected)

    def test_relocate_before(self):
        data = pd.DataFrame({'a': [1],
                             'b': [1],
                             'c': [1],
                             'd': ['a'],
                             'e': ['a'],
                             'f': ['a']})
        expected = pd.DataFrame({'a': [1],
                                 'f': ['a'],
                                 'b': [1],
                                 'c': [1],
                                 'd': ['a'],
                                 'e': ['a']})
        pd.testing.assert_frame_equal(relocate(data, 'f', before='b'), expected)

    def test_relocate_after(self):
        data = pd.DataFrame({'a': [1],
                             'b': [1],
                             'c': [1],
                             'd': ['a'],
                             'e': ['a'],
                             'f': ['a']})
        expected = pd.DataFrame({'b': [1],
                                 'c': [1],
                                 'a': [1],
                                 'd': ['a'],
                                 'e': ['a'],
                                 'f': ['a']})
        pd.testing.assert_frame_equal(relocate(data, 'a', after='c'), expected)


if __name__ == '__main__':
    unittest.main()