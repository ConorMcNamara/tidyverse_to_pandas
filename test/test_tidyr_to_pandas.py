import pandas as pd
import numpy as np
import tidyr_to_pandas
import unittest


class TestTidyrToPandas(unittest.TestCase):

    def test_spread_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, tidyr_to_pandas.spread, data, 'key', 'value')

    def test_spread_DF_result(self):
        df = pd.DataFrame({'row': [1, 1, 1, 51, 51, 51],
                           'var': ['Sepal.length', 'Species', 'species_num', 'Sepal.length', 'Species', 'species_num'],
                           'value': [5.1, 'setosa', 1, 7, 'versicolor', 2]})
        actual = tidyr_to_pandas.spread(df, 'var', 'value')
        expected = pd.DataFrame({'row': [1, 51],
                                 'Sepal.length': [5.1, 7],
                                 'Species': ['setosa', 'versicolor'],
                                 'species_num': [1, 2]})
        actual = actual.apply(pd.to_numeric, errors='ignore')
        pd.testing.assert_frame_equal(actual, expected)

    def test_gather_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, tidyr_to_pandas.gather, data, 'key', 'value', ['a', 'b'])

    def test_gather_DF_result(self):
        df = pd.DataFrame({'names': ['Wilbur', 'Petunia', 'Gregory'],
                           'a': [67, 80, 64],
                           'b': [56, 90, 50]})
        actual = tidyr_to_pandas.gather(df, 'Treatment', 'Heart Rate', ['a', 'b'])
        expected = pd.DataFrame({'names': ['Wilbur', 'Petunia', 'Gregory', 'Wilbur', 'Petunia', 'Gregory'],
                                 'Treatment': ['a', 'a', 'a', 'b', 'b', 'b'],
                                 'Heart Rate': [67, 80, 64, 56, 90, 50]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, tidyr_to_pandas.separate, data, 'X', ['a', 'b'])

    def test_separate_fillLeft_result(self):
        series = pd.DataFrame({'X': ['a', 'a b', 'a b c', 'NA']})
        actual = tidyr_to_pandas.separate(series, 'X', ['A', 'B'], sep=' ', remove=True, extra='drop', fill='left')
        expected = pd.DataFrame({'A': ['NA', 'a', 'a', 'NA'],
                                 'B': ['a', 'b', 'b', 'NA']})
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_fillRight_result(self):
        series = pd.DataFrame({'X': ['a', 'a b', 'a b c', 'NA']})
        actual = tidyr_to_pandas.separate(series, 'X', ['A', 'B'], sep=' ', remove=True, extra='warn', fill='right')
        expected = pd.DataFrame({'A': ['a', 'a', 'a', 'NA'],
                                 'B': ['NA', 'b', 'b', 'NA']})
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_extraMerge_result(self):
        series = pd.DataFrame({'X': ['a', 'a b', 'a b c', 'NA']})
        actual = tidyr_to_pandas.separate(series, 'X', ['A', 'B'], sep=' ', remove=True, extra='merge', fill='warn')
        expected = pd.DataFrame({'A': ['a', 'a', 'a', 'NA'],
                                 'B': ['NA', 'b', 'b c', 'NA']})
        pd.testing.assert_frame_equal(actual, expected)

    def test_unite_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, tidyr_to_pandas.unite, data, 'col', 'index_col')

    def test_unite_nonString_exception(self):
        data = pd.DataFrame({'x': [1, 2, 3, 4]})
        self.assertRaises(Exception, tidyr_to_pandas.unite, data, ['ars', 'x'], 'index_col')

    def test_unite_DF_result(self):
        df = pd.DataFrame({'a': ['a', 'b', 'c', 'd'],
                           'b': [1, 2, 3, 4]})
        actual = tidyr_to_pandas.unite(df, 'a-b', ['a', 'b'])
        expected = pd.DataFrame({'a-b': ['a_1', 'b_2', 'c_3', 'd_4']})
        pd.testing.assert_frame_equal(actual, expected)

    def test_extract_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, tidyr_to_pandas.extract, data, 'col', ['arg', 'kwargs'])

    def test_extract_DF_result(self):
        df = pd.DataFrame({'X': ["NA", "a-b", "a-d", "b-c", "d-e"]})
        actual = tidyr_to_pandas.extract(df, 'X', ['A', 'B'], r'([a-d]+)-([a-d]+)')
        expected = pd.DataFrame({'A': [np.nan, 'a', 'a', 'b', np.nan],
                                 'B': [np.nan, 'b', 'd', 'c', np.nan]})
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
