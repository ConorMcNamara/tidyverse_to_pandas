import pandas as pd
import numpy as np
from src.rebase_tidyr_to_pandas import replace_na, drop_na, unite, extract, fill, separate
import unittest


class TestTidyrToPandas(unittest.TestCase):

    # Replace NA

    def test_replaceNA_pandas(self):
        data = pd.DataFrame({'x': [1, 2, np.nan],
                             'y': ['a', np.nan, 'b'],
                             'z': [[i for i in range(1, 6)], np.nan, [i for i in range(10, 21)]]})
        expected = pd.DataFrame({'x': [1., 2., 0.],
                                 'y': ['a', 'Unknown', 'b'],
                                 'z': [[i for i in range(1, 6)], np.nan, [i for i in range(10, 21)]]})
        pd.testing.assert_frame_equal(replace_na(data, {'x': 0, 'y': 'Unknown'}), expected)

    # Drop NA

    def test_dropNA_pandas(self):
        data = pd.DataFrame({'x': [1, 2, np.nan],
                             'y': ["a", np.nan, "b"]})
        expected = pd.DataFrame({'x': [1.], 'y': ['a']})
        pd.testing.assert_frame_equal(drop_na(data), expected)

    def test_dropNA_pandasSpecification(self):
        data = pd.DataFrame({'x': [1, 2, np.nan],
                             'y': ["a", np.nan, "b"]})
        expected = pd.DataFrame({'x': [1., 2.],
                                 'y': ["a", np.nan]})
        pd.testing.assert_frame_equal(drop_na(data, ['x']), expected)

    # Fill

    def test_fill_pandas(self):
        data = pd.DataFrame({"Month": np.arange(1, 13),
                             "Year": [2000] + [np.nan] * 11})
        expected = pd.DataFrame({'Month': np.arange(1, 13),
                                 'Year': [2000.] * 12})
        pd.testing.assert_frame_equal(fill(data, "Year", direction='down'), expected)

    # Unite

    def test_unite_pandas(self):
        data = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                             'y': ['b', np.nan, 'b', np.nan]})
        expected = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                                 'y': ['b', np.nan, 'b', np.nan],
                                 'z': ['a_b', 'a_NA', 'NA_b', 'NA_NA']})
        pd.testing.assert_frame_equal(unite(data, "z", ['x', 'y'], remove=False), expected)

    def test_unite_pandasNARM(self):
        data = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                             'y': ['b', np.nan, 'b', np.nan]})
        expected = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                                 'y': ['b', np.nan, 'b', np.nan],
                                 'z': ['a_b', 'a', 'b', '']})
        pd.testing.assert_frame_equal(unite(data, 'z', ['x', 'y'], na_rm=True, remove=False), expected)

    # Extract

    def test_extract_pandasMatch(self):
        data = pd.DataFrame({'x': [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({'A': [np.nan, "a", "a", "b", "d"]})
        pd.testing.assert_frame_equal(extract(data, "x", ["A"]), expected)

    def test_extract_pandasMultipleMatch(self):
        data = pd.DataFrame({'x': [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({'A': [np.nan, "a", "a", "b", "d"],
                                 'B': [np.nan, "b", "d", "c", "e"]})
        pd.testing.assert_frame_equal(extract(data, "x", ['A', 'B'], "([a-zA-Z0-9]+)-([a-zA-Z0-9]+)"), expected)

    def test_extract_pandasNoMatchNA(self):
        data = pd.DataFrame({'x': [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({'A': [np.nan, "a", "a", "b", np.nan],
                                 'B': [np.nan, "b", "d", "c", np.nan]})
        pd.testing.assert_frame_equal(extract(data, "x", ['A', 'B'], "([a-d]+)-([a-d]+)"), expected)

    # Separate

    def test_separate_pandas(self):
        data = pd.DataFrame({'x': [np.nan, 'a.b', 'a.d', "b.c"]})
        expected = pd.DataFrame({'A': [np.nan, 'a', 'a', 'b'],
                                 'B': [np.nan, 'b', 'd', 'c']})
        pd.testing.assert_frame_equal(separate(data, "x", ['A', "B"], ".", remove=True), expected)

    def test_separate_pandasNA(self):
        data = pd.DataFrame({'x': [np.nan, 'a.b', 'a.d', "b.c"]})
        expected = pd.DataFrame({'B': [np.nan, 'b', 'd', 'c']})
        pd.testing.assert_frame_equal(separate(data, "x", ["NA", "B"], ".", remove=True), expected)

    def test_separate_fill(self):
        data = pd.DataFrame({'x': ['a', 'a b', 'a b c', np.nan]})
        expected = pd.DataFrame({'A': ['a', 'a', 'a', np.nan],
                                 'B': [np.nan, 'b', 'b', np.nan]})
        pd.testing.assert_frame_equal(separate(data, "x", ['A', "B"], " "), expected)

    def test_separate_fillLeft(self):
        data = pd.DataFrame({'x': ['a', 'a b', 'a b c', np.nan]})
        expected = pd.DataFrame({'A': [np.nan, 'a', 'a', np.nan],
                                 'B': ['a', 'b', 'b c', np.nan]})
        pd.testing.assert_frame_equal(separate(data, "x", ['A', 'B'], " ", fill='left', extra='merge'), expected)

    def test_separate_allThree(self):
        data = pd.DataFrame({'x': ['a', 'a b', 'a b c', np.nan]})
        expected = pd.DataFrame({'A': ['a', 'a', 'a', np.nan],
                                 'B': [np.nan, 'b', 'b', np.nan],
                                 'C': [np.nan, np.nan, 'c', np.nan]})
        pd.testing.assert_frame_equal(separate(data, "x", ['A', 'B', 'C'], " "), expected)


if __name__ == '__main__':
    unittest.main()
