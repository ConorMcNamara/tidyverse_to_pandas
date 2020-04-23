import pandas as pd
import numpy as np
from src.tidyr_to_pandas import replace_na, drop_na, unite, extract, fill, separate, pivot_longer, pivot_wider,\
complete, unnest_longer
import unittest


class TestTidyrToPandas(unittest.TestCase):

    # Pivot Longer
    def test_pivotLonger_pandas(self):
        religion = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\religion.csv")
        pivot_religion = pivot_longer(religion, "-religion", names_to='income', values_to="count")
        assert pivot_religion.shape == (180, 3)

    def test_pivotLonger_pandas_valuesDrop(self):
        billboard = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\billboard.csv")
        pivot_billboard = pivot_longer(billboard, cols="wk1:wk76", names_to="week", names_prefix="wk",
                                       values_to="rank", values_drop_na=True)
        assert pivot_billboard.shape == (5307, 5)

    def test_pivotLonger_pandas_namesPattern(self):
        who = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\who.csv")
        pivot_who = pivot_longer(who, cols="new_sp_m014:newrel_f65", names_to=["diagnosis", "gender", "age"],
                                 names_pattern="new_?(.*)_(.)(.*)", values_to="count", values_drop_na=False)
        assert pivot_who.shape == (405440, 8)

    # Pivot Wider
    def test_pivotWider_pandas(self):
        fish_encounters = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\fish_encounters.csv")
        pivot_fish = pivot_wider(fish_encounters, names_from='station', values_from='seen')
        expected = pd.DataFrame({'fish': ['4842', '4843', '4844', '4845', '4847'],
                                 'Release': [1.] * 5,
                                 'I80_1': [1.] * 5,
                                 'Lisbon': [1.] * 5,
                                 'Rstr': [1.] * 4 + [np.nan],
                                 'BaseTD': [1.] * 4 + [np.nan],
                                 'BCE': [1.] * 3 + [np.nan] * 2,
                                 'BCW': [1.] * 3 + [np.nan] * 2,
                                 'BCE2': [1.] * 3 + [np.nan] * 2,
                                 'BCW2': [1.] * 3 + [np.nan] * 2,
                                 'MAE': [1.] * 3 + [np.nan] * 2,
                                 'MAW': [1.] * 3 + [np.nan] * 2})
        pd.testing.assert_frame_equal(pivot_fish.head(), expected)

    def test_pivotWider_fillNA_pandas(self):
        fish_encounters = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\fish_encounters.csv")
        pivot_fish = pivot_wider(fish_encounters, names_from='station', values_from='seen', values_fill={'seen': 0})
        expected = pd.DataFrame({'fish': ['4842', '4843', '4844', '4845', '4847'],
                                 'Release': [1.] * 5,
                                 'I80_1': [1.] * 5,
                                 'Lisbon': [1.] * 5,
                                 'Rstr': [1.] * 4 + [0.],
                                 'BaseTD': [1.] * 4 + [0.],
                                 'BCE': [1.] * 3 + [0.] * 2,
                                 'BCW': [1.] * 3 + [0.] * 2,
                                 'BCE2': [1.] * 3 + [0.] * 2,
                                 'BCW2': [1.] * 3 + [0.] * 2,
                                 'MAE': [1.] * 3 + [0.] * 2,
                                 'MAW': [1.] * 3 + [0.] * 2})
        pd.testing.assert_frame_equal(pivot_fish.head(), expected)

    # Unnest Longer

    def test_unnestLonger_pandas(self):
        data = pd.DataFrame({'x': [1, 2],
                             'y': [{'a': 1, 'b': 2}, {'a': 10, 'b': 11, 'c': 12}]})
        expected = pd.DataFrame({'x': [1, 1, 2, 2, 2],
                                 'y': [1., 2., 10., 11., 12.],
                                 'y_id': ['a', 'b', 'a', 'b', 'c']})
        actual = unnest_longer(data, 'y')
        pd.testing.assert_frame_equal(actual, expected)

    def test_unnestLonger_pandasAgain(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [np.nan, [1, 2, 3], [4, 5]]})
        expected = pd.DataFrame({'x': [1, 2, 2, 2, 3, 3],
                                 'y': [np.nan, 1, 2, 3, 4, 5]})
        actual = unnest_longer(df, 'y', simplify=True)
        pd.testing.assert_frame_equal(actual, expected)


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

    # Compete

    def test_complete_pandas(self):
        data = pd.DataFrame({'group': [1, 2, 1],
                             'item_id': [1, 2, 2],
                             'item_name': ['a', 'b', 'b'],
                             'value1': [1, 2, 3],
                             'value2': [4, 5, 6]})
        expected = pd.DataFrame({'group': [1, 1, 2, 2],
                                 'item_id': [1, 2, 1, 2],
                                 'item_name': ['a', 'b', 'a', 'b'],
                                 'value1': [1, 3, np.nan, 2],
                                 'value2': [4, 6, np.nan, 5]})
        actual = complete(data, ["group", "nesting(item_id, item_name)"])
        pd.testing.assert_frame_equal(actual, expected)

    def test_compete_pandasFill(self):
        data = pd.DataFrame({'group': [1, 2, 1],
                             'item_id': [1, 2, 2],
                             'item_name': ['a', 'b', 'b'],
                             'value1': [1, 2, 3],
                             'value2': [4, 5, 6]})
        expected = pd.DataFrame({'group': [1, 1, 2, 2],
                                 'item_id': [1, 2, 1, 2],
                                 'item_name': ['a', 'b', 'a', 'b'],
                                 'value1': [1., 3., 0., 2.],
                                 'value2': [4, 6, np.nan, 5]})
        actual = complete(data, ["group", "nesting(item_id, item_name)"], fill={'value1': 0})
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
