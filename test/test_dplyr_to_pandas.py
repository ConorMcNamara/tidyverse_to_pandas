import pandas as pd
from src.rebase_dplyr_to_pandas import arrange, distinct, filter, pull
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
        expected = pd.DataFrame({'name': {15: 'Jabba Desilijic Tiure'}, 'height': {15: 175.0}, 'mass': {15: 1358.0},
                                 'hair_color': {15: np.nan}, 'skin_color': {15: 'green-tan, brown'}, 'eye_color': {15: 'orange'},
                                 'birth_year': {15: 600.0}, 'sex': {15: 'hermaphroditic'}, 'gender': {15: 'masculine'},
                                 'homeworld': {15: 'Nal Hutta'}, 'species': {15: 'Hutt'}})
        expected['hair_color'] = expected['hair_color'].astype('object')
        pd.testing.assert_frame_equal(expected, filter(starwars, 'mass > 1000'))

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