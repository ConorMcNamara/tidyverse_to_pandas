import pandas as pd
import numpy as np
from src.tidyr_to_pandas import replace_na, drop_na, unite, extract, fill, separate, pivot_longer, pivot_wider,\
complete, unnest_longer, unnest_wider, nest, unnest, chop, unchop
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
        expected = pd.DataFrame({'fish': [4842, 4843, 4844, 4845, 4847],
                                 'Release': [1.] * 5,
                                 'I80_1': [1.] * 5,
                                 'Lisbon': [1.] * 5,
                                 'Rstr': [1.] * 4 + [np.nan],
                                 'Base_TD': [1.] * 4 + [np.nan],
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
        expected = pd.DataFrame({'fish': [4842, 4843, 4844, 4845, 4847],
                                 'Release': [1.] * 5,
                                 'I80_1': [1.] * 5,
                                 'Lisbon': [1.] * 5,
                                 'Rstr': [1.] * 4 + [0.],
                                 'Base_TD': [1.] * 4 + [0.],
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

    # Unnest Wider

    def test_unnestWider_pandasDict(self):
        df = pd.DataFrame({'character': ["Toothless", "Dory"],
                           'metadata': [{
                               'species': "dragon",
                               'color': "black",
                               'films': ["How to Train Your Dragon", "How to Train Your Dragon 2",
                                         "How to Train Your Dragon: The Hidden World"]
                           }, {
                               'species': "clownfish",
                               'color': "blue",
                               'films': ["Finding Nemo", "Finding Dory"]}]})
        expected = pd.DataFrame({'character': ['Toothless', 'Dory'],
                                 'species': ['dragon', 'clownfish'],
                                 'color': ['black', 'blue'],
                                 'films': [["How to Train Your Dragon", "How to Train Your Dragon 2",
                                            "How to Train Your Dragon: The Hidden World"],
                                           ["Finding Nemo", "Finding Dory"]]
                                 })
        actual = unnest_wider(df, 'metadata')
        pd.testing.assert_frame_equal(actual, expected)

    def test_unnestWider_pandasList(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [np.nan, [1, 2, 3], [4, 5]]})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y0': [np.nan, 1.0, 4.0],
                                 'y1': [np.nan, 2.0, 5.0],
                                 'y2': [np.nan, 3.0, np.nan]})
        actual = unnest_wider(df, ['y'])
        pd.testing.assert_frame_equal(actual, expected)

    # Nest

    def test_nest_pandas(self):
        data = pd.DataFrame({'x': [1, 1, 1, 2, 2, 3], 'y': np.arange(1, 7), 'z': np.arange(6, 0, -1)})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'data': [{'y': {0: 1, 1: 2, 2: 3},
                                           'z': {0: 6, 1: 5, 2: 4}},
                                          {'y': {3: 4, 4: 5},
                                           'z': {3: 3, 4: 2}},
                                          {'y': {5: 6},
                                           'z': {5: 1}}]})
        actual = nest(data, ['y', 'z'])
        pd.testing.assert_frame_equal(actual, expected)

    def test_nest_pandasIris(self):
        data = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\iris.csv")
        actual = nest(data, '-Species')
        assert actual.shape == (3, 2)

    # Unnest

    def test_unnest_pandasIris(self):
        data = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\iris.csv")
        nested_data = nest(data, '-Species')
        actual = unnest(nested_data, 'data')
        data = data[['Species'] + list(data.columns.difference(['Species']))]
        pd.testing.assert_frame_equal(actual, data)

    def test_unnest_multipleColumnsPandasIris(self):
        data = pd.read_csv("C:\\Users\\conor\\Documents\\tidyverse_to_pandas\\data\\iris.csv")
        nested_data = nest(data, {'petal': ['Petal.Length', 'Petal.Width'], 'sepal': ['Sepal.Length', 'Sepal.Width']})
        actual = unnest(nested_data, ['petal', 'sepal'])
        data = data[['Species', 'Petal.Length', 'Petal.Width', 'Sepal.Length', 'Sepal.Width']]
        pd.testing.assert_frame_equal(actual, data)

    # Chop

    def test_chop_pandas(self):
        data = pd.DataFrame({'x': [1, 1, 1, 2, 2, 3], 'y': np.arange(1, 7), 'z': np.arange(6, 0, -1)})
        expected = pd.DataFrame({'x': [1, 2, 3],
                                 'y': [[1, 2, 3], [4, 5], [6]],
                                 'z': [[6, 5, 4], [3, 2], [1]]})
        actual = chop(data, ['y', 'z'])
        pd.testing.assert_frame_equal(actual, expected)

    # Unchop

    def test_unchop_pandas(self):
        data = pd.DataFrame({'x': [1, 2, 3, 4],
                             'y': [[], [1], [1, 2], [1, 2, 3]]})
        actual = unchop(data, 'y')
        expected = pd.DataFrame({'x': [2, 3, 3, 4, 4, 4],
                                 'y': [1, 1, 2, 1, 2, 3]})
        pd.testing.assert_frame_equal(actual, expected)

    # Unite

    def test_unite_pandas(self):
        data = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                             'y': ['b', np.nan, 'b', np.nan]})
        expected = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                                 'y': ['b', np.nan, 'b', np.nan],
                                 'z': ['a_b', 'a_NA', 'NA_b', 'NA_NA']})
        actual = unite(data, "z", ['x', 'y'], remove=False)
        pd.testing.assert_frame_equal(actual, expected)

    def test_unite_pandasNARM(self):
        data = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                             'y': ['b', np.nan, 'b', np.nan]})
        expected = pd.DataFrame({'x': ['a', 'a', np.nan, np.nan],
                                 'y': ['b', np.nan, 'b', np.nan],
                                 'z': ['a_b', 'a', 'b', '']})
        actual = unite(data, 'z', ['x', 'y'], na_rm=True, remove=False)
        pd.testing.assert_frame_equal(actual, expected)

    # Extract

    def test_extract_pandasMatch(self):
        data = pd.DataFrame({'x': [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({'A': [np.nan, "a", "a", "b", "d"]})
        actual = extract(data, "x", ["A"])
        pd.testing.assert_frame_equal(actual, expected)

    def test_extract_pandasMultipleMatch(self):
        data = pd.DataFrame({'x': [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({'A': [np.nan, "a", "a", "b", "d"],
                                 'B': [np.nan, "b", "d", "c", "e"]})
        actual = extract(data, "x", ['A', 'B'], "([a-zA-Z0-9]+)-([a-zA-Z0-9]+)")
        pd.testing.assert_frame_equal(actual, expected)

    def test_extract_pandasNoMatchNA(self):
        data = pd.DataFrame({'x': [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({'A': [np.nan, "a", "a", "b", np.nan],
                                 'B': [np.nan, "b", "d", "c", np.nan]})
        actual = extract(data, "x", ['A', 'B'], "([a-d]+)-([a-d]+)")
        pd.testing.assert_frame_equal(actual, expected)

    # Separate

    def test_separate_pandas(self):
        data = pd.DataFrame({'x': [np.nan, 'a.b', 'a.d', "b.c"]})
        expected = pd.DataFrame({'A': [np.nan, 'a', 'a', 'b'],
                                 'B': [np.nan, 'b', 'd', 'c']})
        actual = separate(data, "x", ['A', "B"], ".", remove=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_pandasNA(self):
        data = pd.DataFrame({'x': [np.nan, 'a.b', 'a.d', "b.c"]})
        expected = pd.DataFrame({'B': [np.nan, 'b', 'd', 'c']})
        actual = separate(data, "x", ["NA", "B"], ".", remove=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_fill(self):
        data = pd.DataFrame({'x': ['a', 'a b', 'a b c', np.nan]})
        expected = pd.DataFrame({'A': ['a', 'a', 'a', np.nan],
                                 'B': [np.nan, 'b', 'b', np.nan]})
        actual = separate(data, "x", ['A', "B"], " ")
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_fillLeft(self):
        data = pd.DataFrame({'x': ['a', 'a b', 'a b c', np.nan]})
        expected = pd.DataFrame({'A': [np.nan, 'a', 'a', np.nan],
                                 'B': ['a', 'b', 'b c', np.nan]})
        actual = separate(data, "x", ['A', 'B'], " ", fill='left', extra='merge')
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_allThree(self):
        data = pd.DataFrame({'x': ['a', 'a b', 'a b c', np.nan]})
        expected = pd.DataFrame({'A': ['a', 'a', 'a', np.nan],
                                 'B': [np.nan, 'b', 'b', np.nan],
                                 'C': [np.nan, np.nan, 'c', np.nan]})
        actual = separate(data, "x", ['A', 'B', 'C'], " ")
        pd.testing.assert_frame_equal(actual, expected)

        # Replace NA

    def test_replaceNA_pandas(self):
        data = pd.DataFrame({'x': [1, 2, np.nan],
                             'y': ['a', np.nan, 'b'],
                             'z': [[i for i in range(1, 6)], np.nan, [i for i in range(10, 21)]]})
        expected = pd.DataFrame({'x': [1., 2., 0.],
                                 'y': ['a', 'Unknown', 'b'],
                                 'z': [[i for i in range(1, 6)], np.nan, [i for i in range(10, 21)]]})
        actual = replace_na(data, {'x': 0, 'y': 'Unknown'})
        pd.testing.assert_frame_equal(actual, expected)

        # Drop NA

    def test_dropNA_pandas(self):
        data = pd.DataFrame({'x': [1, 2, np.nan],
                             'y': ["a", np.nan, "b"]})
        expected = pd.DataFrame({'x': [1.], 'y': ['a']})
        actual = drop_na(data)
        pd.testing.assert_frame_equal(actual, expected)

    def test_dropNA_pandasSpecification(self):
        data = pd.DataFrame({'x': [1, 2, np.nan],
                             'y': ["a", np.nan, "b"]})
        expected = pd.DataFrame({'x': [1., 2.],
                                 'y': ["a", np.nan]})
        actual = drop_na(data, ['x'])
        pd.testing.assert_frame_equal(actual, expected)

        # Fill

    def test_fill_pandas(self):
        data = pd.DataFrame({"Month": np.arange(1, 13),
                             "Year": [2000] + [np.nan] * 11})
        expected = pd.DataFrame({'Month': np.arange(1, 13),
                                 'Year': [2000.] * 12})
        actual = fill(data, "Year", direction='down')
        pd.testing.assert_frame_equal(actual, expected)

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
