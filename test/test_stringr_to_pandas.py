import unittest
import pytest
import pandas as pd
import numpy as np
from src.stringr_to_pandas import str_length, str_sub, str_detect, str_count


class TestStringrToPandas(unittest.TestCase):

    # String Length
    def test_strLength_string(self):
        string = 'abcd'
        assert str_length(string) == 4

    def test_strLength_list(self):
        string = ['ab', 'cdef']
        assert str_length(string) == [2, 4]

    def test_strLength_array(self):
        string = np.array(['abc', 'defg'])
        np.testing.assert_array_equal(str_length(string), np.array([3, 4]))

    def test_strLength_series(self):
        string = pd.Series(['abcd', 'efg'])
        pd.testing.assert_series_equal(str_length(string), pd.Series([4, 3]))

    # String Sub
    def test_strSub_string(self):
        string = 'abcd'
        assert str_sub(string, 0, 3) == 'abc'

    def test_strSub_list(self):
        string = ["abcdef", "ghifjk"]
        assert str_sub(string, 2, 3) == ['c', 'i']

    def test_strSub_array(self):
        string = np.array(["abcdef", "ghifjk"])
        np.testing.assert_array_equal(str_sub(string, 1, -1), np.array(['bcde', 'hifj']))

    def test_strSub_series(self):
        string = pd.Series(["abcdef", "ghifjk"])
        pd.testing.assert_series_equal(str_sub(string, 2, -1), pd.Series(['cde', 'ifj']))

    # String Detect
    def test_strDetect_string(self):
        string = 'video'
        assert str_detect(string, '[aeiou]') is True

    def test_strDetect_list(self):
        x = ["why", "video", "cross", "extra", "deal", "authority"]
        assert str_detect(x, '[aeiou]') == [False, True, True, True, True, True]

    def test_strDetect_array(self):
        x = np.array(["why", "video", "cross", "extra", "deal", "authority"])
        np.testing.assert_array_equal(str_detect(x, '[aeiou]', negate=True), np.array([True, False, False, False, False, False]))

    def test_strDetect_series(self):
        x = pd.Series(["why", "video", "cross", "extra", "deal", "authority"])
        pd.testing.assert_series_equal(str_detect(x, '[aeiou]'), pd.Series([False, True, True, True, True, True]))

    # String Count
    def test_strCount_string(self):
        string = 'video'
        assert str_count(string, '[aeiou]') == 3

    def test_strCount_list(self):
        fruit = ["apple", "banana", "pear", "pineapple"]
        assert str_count(fruit, 'a') == [1, 3, 1, 1]

    def test_strCount_array(self):
        fruit = np.array(["apple", "banana", "pear", "pineapple"])
        np.testing.assert_array_equal(str_count(fruit, 'e'), np.array([1, 0, 1, 2]))

    def test_strCount_series(self):
        fruit = pd.Series(["apple", "banana", "pear", "pineapple"])
        pd.testing.assert_series_equal(str_count(fruit, ['a', 'b', 'p', 'p']), pd.Series([1, 1, 1, 3]))