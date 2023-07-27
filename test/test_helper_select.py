from src.helper_select import everything, last_col
import pandas as pd
import numpy as np
import unittest
import pytest


class TestHelperSelect(unittest.TestCase):

    def test_everything(self) -> None:
        cars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\mtcars.csv')
        cars = cars.rename({'Unnamed: 0': 'car'}, axis=1)
        expected = ['car', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
        np.testing.assert_array_equal(everything(cars), expected)

    def test_lastCol_noOffset(self) -> None:
        cars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\mtcars.csv')
        cars = cars.rename({'Unnamed: 0': 'car'}, axis=1)
        assert last_col(cars) == 'carb'

    def test_lastCol_offset(self) -> None:
        cars = pd.read_csv('C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\mtcars.csv')
        cars = cars.rename({'Unnamed: 0': 'car'}, axis=1)
        assert last_col(cars, 2) == 'am'
