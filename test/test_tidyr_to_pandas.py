from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tidyverse_to_pandas.tidyr_to_pandas import (
    chop,
    complete,
    drop_na,
    extract,
    fill,
    nest,
    pivot_longer,
    pivot_wider,
    replace_na,
    separate,
    unchop,
    unite,
    unnest,
    unnest_longer,
    unnest_wider,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestTidyrToPandas:
    # Pivot Longer
    def test_pivotLonger_pandas(self) -> None:
        religion = pd.read_csv(DATA_DIR / "religion.csv")
        pivot_religion = pivot_longer(religion, "-religion", names_to="income", values_to="count")
        assert pivot_religion.shape == (180, 3)

    def test_pivotLonger_pandas_valuesDrop(self) -> None:
        billboard = pd.read_csv(DATA_DIR / "billboard.csv")
        pivot_billboard = pivot_longer(
            billboard,
            cols="wk1:wk76",
            names_to="week",
            names_prefix="wk",
            values_to="rank",
            values_drop_na=True,
        )
        assert pivot_billboard.shape == (5307, 5)

    def test_pivotLonger_pandas_namesPattern(self) -> None:
        who = pd.read_csv(DATA_DIR / "who.csv")
        pivot_who = pivot_longer(
            who,
            cols="new_sp_m014:newrel_f65",
            names_to=["diagnosis", "gender", "age"],
            names_pattern="new_?(.*)_(.)(.*)",
            values_to="count",
            values_drop_na=False,
        )
        assert pivot_who.shape == (405440, 8)

    # Pivot Wider
    def test_pivotWider_pandas(self) -> None:
        fish_encounters = pd.read_csv(DATA_DIR / "fish_encounters.csv")
        pivot_fish = pivot_wider(fish_encounters, names_from="station", values_from="seen")
        expected = pd.DataFrame(
            {
                "fish": [4842, 4843, 4844, 4845, 4847],
                "Release": [1.0] * 5,
                "I80_1": [1.0] * 5,
                "Lisbon": [1.0] * 5,
                "Rstr": [1.0] * 4 + [np.nan],
                "Base_TD": [1.0] * 4 + [np.nan],
                "BCE": [1.0] * 3 + [np.nan] * 2,
                "BCW": [1.0] * 3 + [np.nan] * 2,
                "BCE2": [1.0] * 3 + [np.nan] * 2,
                "BCW2": [1.0] * 3 + [np.nan] * 2,
                "MAE": [1.0] * 3 + [np.nan] * 2,
                "MAW": [1.0] * 3 + [np.nan] * 2,
            }
        )
        pd.testing.assert_frame_equal(pivot_fish.head(), expected)

    def test_pivotWider_fillNA_pandas(self) -> None:
        fish_encounters = pd.read_csv(DATA_DIR / "fish_encounters.csv")
        pivot_fish = pivot_wider(
            fish_encounters,
            names_from="station",
            values_from="seen",
            values_fill={"seen": 0},
        )
        expected = pd.DataFrame(
            {
                "fish": [4842, 4843, 4844, 4845, 4847],
                "Release": [1.0] * 5,
                "I80_1": [1.0] * 5,
                "Lisbon": [1.0] * 5,
                "Rstr": [1.0] * 4 + [0.0],
                "Base_TD": [1.0] * 4 + [0.0],
                "BCE": [1.0] * 3 + [0.0] * 2,
                "BCW": [1.0] * 3 + [0.0] * 2,
                "BCE2": [1.0] * 3 + [0.0] * 2,
                "BCW2": [1.0] * 3 + [0.0] * 2,
                "MAE": [1.0] * 3 + [0.0] * 2,
                "MAW": [1.0] * 3 + [0.0] * 2,
            }
        )
        pd.testing.assert_frame_equal(pivot_fish.head(), expected)

    # Unnest Longer

    def test_unnestLonger_pandas(self) -> None:
        data = pd.DataFrame({"x": [1, 2], "y": [{"a": 1, "b": 2}, {"a": 10, "b": 11, "c": 12}]})
        expected = pd.DataFrame(
            {
                "x": [1, 1, 2, 2, 2],
                "y": [1, 2, 10, 11, 12],
                "y_id": ["a", "b", "a", "b", "c"],
            }
        )
        actual = unnest_longer(data, "y")
        pd.testing.assert_frame_equal(actual, expected)

    def test_unnestLonger_pandasAgain(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [np.nan, [1, 2, 3], [4, 5]]})
        expected = pd.DataFrame({"x": [1, 2, 2, 2, 3, 3], "y": [np.nan, 1, 2, 3, 4, 5]})
        actual = unnest_longer(df, "y", simplify=True)
        pd.testing.assert_frame_equal(actual, expected)

    # Unnest Wider

    def test_unnestWider_pandasDict(self) -> None:
        df = pd.DataFrame(
            {
                "character": ["Toothless", "Dory"],
                "metadata": [
                    {
                        "species": "dragon",
                        "color": "black",
                        "films": [
                            "How to Train Your Dragon",
                            "How to Train Your Dragon 2",
                            "How to Train Your Dragon: The Hidden World",
                        ],
                    },
                    {
                        "species": "clownfish",
                        "color": "blue",
                        "films": ["Finding Nemo", "Finding Dory"],
                    },
                ],
            }
        )
        expected = pd.DataFrame(
            {
                "character": ["Toothless", "Dory"],
                "species": ["dragon", "clownfish"],
                "color": ["black", "blue"],
                "films": [
                    [
                        "How to Train Your Dragon",
                        "How to Train Your Dragon 2",
                        "How to Train Your Dragon: The Hidden World",
                    ],
                    ["Finding Nemo", "Finding Dory"],
                ],
            }
        )
        actual = unnest_wider(df, "metadata")
        pd.testing.assert_frame_equal(actual, expected)

    def test_unnestWider_pandasList(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [np.nan, [1, 2, 3], [4, 5]]})
        expected = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y_1": [np.nan, 1.0, 4.0],
                "y_2": [np.nan, 2.0, 5.0],
                "y_3": [np.nan, 3.0, np.nan],
            }
        )
        actual = unnest_wider(df, ["y"])
        pd.testing.assert_frame_equal(actual, expected)

    # Nest

    def test_nest_pandas(self) -> None:
        data = pd.DataFrame({"x": [1, 1, 1, 2, 2, 3], "y": np.arange(1, 7), "z": np.arange(6, 0, -1)})
        expected = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "data": [
                    {"y": {0: 1, 1: 2, 2: 3}, "z": {0: 6, 1: 5, 2: 4}},
                    {"y": {3: 4, 4: 5}, "z": {3: 3, 4: 2}},
                    {"y": {5: 6}, "z": {5: 1}},
                ],
            }
        )
        actual = nest(data, ["y", "z"])
        pd.testing.assert_frame_equal(actual, expected)

    def test_nest_pandasIris(self) -> None:
        data = pd.read_csv(DATA_DIR / "iris.csv")
        actual = nest(data, "-Species")
        assert actual.shape == (3, 2)

    # Unnest

    def test_unnest_pandasIris(self) -> None:
        data = pd.read_csv(DATA_DIR / "iris.csv")
        nested_data = nest(data, "-Species")
        actual = unnest(nested_data, "data")
        data = data[["Species"] + [c for c in data.columns if c != "Species"]]
        pd.testing.assert_frame_equal(actual, data)

    def test_unnest_multipleColumnsPandasIris(self) -> None:
        data = pd.read_csv(DATA_DIR / "iris.csv")
        nested_data = nest(
            data,
            {
                "petal": ["Petal.Length", "Petal.Width"],
                "sepal": ["Sepal.Length", "Sepal.Width"],
            },
        )
        actual = unnest(nested_data, ["petal", "sepal"])
        data = data[["Species", "Petal.Length", "Petal.Width", "Sepal.Length", "Sepal.Width"]]
        pd.testing.assert_frame_equal(actual, data)

    # Chop

    def test_chop_pandas(self) -> None:
        data = pd.DataFrame({"x": [1, 1, 1, 2, 2, 3], "y": np.arange(1, 7), "z": np.arange(6, 0, -1)})
        expected = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [[1, 2, 3], [4, 5], [6]],
                "z": [[6, 5, 4], [3, 2], [1]],
            }
        )
        actual = chop(data, ["y", "z"])
        pd.testing.assert_frame_equal(actual, expected)

    # Unchop

    def test_unchop_pandas(self) -> None:
        data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [[], [1], [1, 2], [1, 2, 3]]})
        actual = unchop(data, "y")
        expected = pd.DataFrame({"x": [2, 3, 3, 4, 4, 4], "y": [1, 1, 2, 1, 2, 3]})
        pd.testing.assert_frame_equal(actual, expected)

    # Unite

    def test_unite_pandas(self) -> None:
        data = pd.DataFrame({"x": ["a", "a", np.nan, np.nan], "y": ["b", np.nan, "b", np.nan]})
        expected = pd.DataFrame(
            {
                "z": ["a_b", "a_NA", "NA_b", "NA_NA"],
                "x": ["a", "a", np.nan, np.nan],
                "y": ["b", np.nan, "b", np.nan],
            }
        )
        actual = unite(data, "z", ["x", "y"], remove=False)
        pd.testing.assert_frame_equal(actual, expected)

    def test_unite_pandasNARM(self) -> None:
        data = pd.DataFrame({"x": ["a", "a", np.nan, np.nan], "y": ["b", np.nan, "b", np.nan]})
        expected = pd.DataFrame(
            {
                "z": ["a_b", "a", "b", ""],
                "x": ["a", "a", np.nan, np.nan],
                "y": ["b", np.nan, "b", np.nan],
            }
        )
        actual = unite(data, "z", ["x", "y"], na_rm=True, remove=False)
        pd.testing.assert_frame_equal(actual, expected)

    # Extract

    def test_extract_pandasMatch(self) -> None:
        data = pd.DataFrame({"x": [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({"A": [np.nan, "a", "a", "b", "d"]})
        actual = extract(data, "x", ["A"])
        pd.testing.assert_frame_equal(actual, expected)

    def test_extract_pandasMultipleMatch(self) -> None:
        data = pd.DataFrame({"x": [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({"A": [np.nan, "a", "a", "b", "d"], "B": [np.nan, "b", "d", "c", "e"]})
        actual = extract(data, "x", ["A", "B"], "([a-zA-Z0-9]+)-([a-zA-Z0-9]+)")
        pd.testing.assert_frame_equal(actual, expected)

    def test_extract_pandasNoMatchNA(self) -> None:
        data = pd.DataFrame({"x": [np.nan, "a-b", "a-d", "b-c", "d-e"]})
        expected = pd.DataFrame({"A": [np.nan, "a", "a", "b", np.nan], "B": [np.nan, "b", "d", "c", np.nan]})
        actual = extract(data, "x", ["A", "B"], "([a-d]+)-([a-d]+)")
        pd.testing.assert_frame_equal(actual, expected)

    # Separate

    def test_separate_pandas(self) -> None:
        data = pd.DataFrame({"x": [np.nan, "a.b", "a.d", "b.c"]})
        expected = pd.DataFrame({"A": [np.nan, "a", "a", "b"], "B": [np.nan, "b", "d", "c"]})
        actual = separate(data, "x", ["A", "B"], ".", remove=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_pandasNA(self) -> None:
        data = pd.DataFrame({"x": [np.nan, "a.b", "a.d", "b.c"]})
        expected = pd.DataFrame({"B": [np.nan, "b", "d", "c"]})
        actual = separate(data, "x", ["NA", "B"], ".", remove=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_fill(self) -> None:
        data = pd.DataFrame({"x": ["a", "a b", "a b c", np.nan]})
        expected = pd.DataFrame({"A": ["a", "a", "a", np.nan], "B": [np.nan, "b", "b", np.nan]})
        actual = separate(data, "x", ["A", "B"], " ")
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_fillLeft(self) -> None:
        data = pd.DataFrame({"x": ["a", "a b", "a b c", np.nan]})
        expected = pd.DataFrame({"A": [np.nan, "a", "a", np.nan], "B": ["a", "b", "b c", np.nan]})
        actual = separate(data, "x", ["A", "B"], " ", fill="left", extra="merge")
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_allThree(self) -> None:
        data = pd.DataFrame({"x": ["a", "a b", "a b c", np.nan]})
        expected = pd.DataFrame(
            {
                "A": ["a", "a", "a", np.nan],
                "B": [np.nan, "b", "b", np.nan],
                "C": [np.nan, np.nan, "c", np.nan],
            }
        )
        actual = separate(data, "x", ["A", "B", "C"], " ")
        pd.testing.assert_frame_equal(actual, expected)

        # Replace NA

    def test_replaceNA_pandas(self) -> None:
        data = pd.DataFrame(
            {
                "x": [1, 2, np.nan],
                "y": ["a", np.nan, "b"],
                "z": [[i for i in range(1, 6)], np.nan, [i for i in range(10, 21)]],
            }
        )
        expected = pd.DataFrame(
            {
                "x": [1.0, 2.0, 0.0],
                "y": ["a", "Unknown", "b"],
                "z": [[i for i in range(1, 6)], np.nan, [i for i in range(10, 21)]],
            }
        )
        actual = replace_na(data, {"x": 0, "y": "Unknown"})
        pd.testing.assert_frame_equal(actual, expected)

        # Drop NA

    def test_dropNA_pandas(self) -> None:
        data = pd.DataFrame({"x": [1, 2, np.nan], "y": ["a", np.nan, "b"]})
        expected = pd.DataFrame({"x": [1.0], "y": ["a"]})
        actual = drop_na(data)
        pd.testing.assert_frame_equal(actual, expected)

    def test_dropNA_pandasSpecification(self) -> None:
        data = pd.DataFrame({"x": [1, 2, np.nan], "y": ["a", np.nan, "b"]})
        expected = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", np.nan]})
        actual = drop_na(data, ["x"])
        pd.testing.assert_frame_equal(actual, expected)

        # Fill

    def test_fill_pandas(self) -> None:
        data = pd.DataFrame({"Month": np.arange(1, 13), "Year": [2000] + [np.nan] * 11})
        expected = pd.DataFrame({"Month": np.arange(1, 13), "Year": [2000.0] * 12})
        actual = fill(data, "Year", direction="down")
        pd.testing.assert_frame_equal(actual, expected)

    # Complete

    def test_complete_pandas(self) -> None:
        data = pd.DataFrame(
            {
                "group": [1, 2, 1],
                "item_id": [1, 2, 2],
                "item_name": ["a", "b", "b"],
                "value1": [1, 2, 3],
                "value2": [4, 5, 6],
            }
        )
        expected = pd.DataFrame(
            {
                "group": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "item_name": ["a", "b", "a", "b"],
                "value1": [1, 3, np.nan, 2],
                "value2": [4, 6, np.nan, 5],
            }
        )
        actual = complete(data, ["group", "nesting(item_id, item_name)"])
        pd.testing.assert_frame_equal(actual, expected)

    def test_complete_pandasFill(self) -> None:
        data = pd.DataFrame(
            {
                "group": [1, 2, 1],
                "item_id": [1, 2, 2],
                "item_name": ["a", "b", "b"],
                "value1": [1, 2, 3],
                "value2": [4, 5, 6],
            }
        )
        expected = pd.DataFrame(
            {
                "group": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "item_name": ["a", "b", "a", "b"],
                "value1": [1.0, 3.0, 0.0, 2.0],
                "value2": [4, 6, np.nan, 5],
            }
        )
        actual = complete(data, ["group", "nesting(item_id, item_name)"], fill={"value1": 0})
        pd.testing.assert_frame_equal(actual, expected)


class TestTidyrRegressions:
    # Pivot Longer
    def test_pivotLonger_dropNAOnlyValueColumn(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "x_1": [1, np.nan], "x_2": [3, 4], "other": [np.nan, 5]})
        actual = pivot_longer(df, ["x_1", "x_2"], values_drop_na=True)
        expected = pd.DataFrame(
            {"id": [1, 1, 2], "other": [np.nan, np.nan, 5.0], "name": ["x_1", "x_2", "x_2"], "value": [1.0, 3.0, 4.0]}
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_pivotLonger_keepsNAByDefault(self) -> None:
        df = pd.DataFrame({"id": [1], "a": [np.nan], "b": [2.0]})
        assert len(pivot_longer(df, ["a", "b"])) == 2

    def test_pivotLonger_namesPrefixAnchored(self) -> None:
        df = pd.DataFrame({"id": [1], "x_x1": [1], "wk1": [2]})
        assert pivot_longer(df, ["x_x1"], names_prefix="x")["name"].tolist() == ["_x1"]
        assert pivot_longer(df, ["wk1"], names_prefix="^wk")["name"].tolist() == ["1"]

    def test_pivotLonger_namesPrefixWithNamesSep(self) -> None:
        df = pd.DataFrame({"id": [1], "wk_a_1": [5]})
        actual = pivot_longer(df, ["wk_a_1"], names_to=["letter", "num"], names_prefix="wk_", names_sep="_")
        assert actual.columns.tolist() == ["id", "letter", "num", "value"]
        assert actual.iloc[0].tolist() == [1, "a", "1", 5]

    def test_pivotLonger_namesPtypesSingleName(self) -> None:
        df = pd.DataFrame({"id": [1], "1": [5], "2": [6]})
        actual = pivot_longer(df, ["1", "2"], names_ptypes=[int])
        assert actual["name"].tolist() == [1, 2]

    def test_pivotLonger_keepsIdOrderAndRowOrder(self) -> None:
        df = pd.DataFrame({"z": [1, 2], "a": [3, 4], "p": [5, 6], "q": [7, 8]})
        actual = pivot_longer(df, ["p", "q"])
        assert actual.columns.tolist() == ["z", "a", "name", "value"]
        assert actual["value"].tolist() == [5, 7, 6, 8]

    # Pivot Wider
    def test_pivotWider_namesPrefix(self) -> None:
        df = pd.DataFrame({"id": [1, 1], "k": [1, 2], "v": [3, 4]})
        actual = pivot_wider(df, names_from="k", values_from="v", names_prefix="p_")
        assert actual.columns.tolist() == ["id", "p_1", "p_2"]

    def test_pivotWider_valuesFn(self) -> None:
        df = pd.DataFrame({"id": [1, 1], "k": ["a", "a"], "v": [1, 4]})
        actual = pivot_wider(df, names_from="k", values_from="v", values_fn="sum")
        assert actual["a"].tolist() == [5]

    def test_pivotWider_duplicatesNotAveraged(self) -> None:
        df = pd.DataFrame({"g": [1, 1], "id": [1, 1], "k": ["a", "a"], "v": [1, 4]})
        with pytest.warns(UserWarning, match="not uniquely identified"):
            actual = pivot_wider(df, names_from="k", values_from="v")
        assert actual["a"].tolist() == [[1, 4]]

    def test_pivotWider_multipleValuesFrom(self) -> None:
        df = pd.DataFrame({"id": [1, 1], "k": ["a", "b"], "v1": [1, 2], "v2": [3, 4]})
        actual = pivot_wider(df, names_from="k", values_from=["v1", "v2"])
        expected = pd.DataFrame({"id": [1], "v1_a": [1], "v1_b": [2], "v2_a": [3], "v2_b": [4]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_pivotWider_idColumnOrder(self) -> None:
        df = pd.DataFrame({"id": [2, 1], "g": ["x", "y"], "k": ["a", "a"], "v": [1, 2]})
        actual = pivot_wider(df, names_from="k", values_from="v")
        assert actual.columns.tolist() == ["id", "g", "a"]
        assert actual["id"].tolist() == [2, 1]

    # Unnest Longer
    def test_unnestLonger_multipleColumnsAligned(self) -> None:
        df = pd.DataFrame({"g": [1, 2], "x": [[1, 2], [3, 4]], "y": [["a", "b"], ["c", "d"]]})
        actual = unnest_longer(df, ["x", "y"])
        expected = pd.DataFrame({"g": [1, 1, 2, 2], "x": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_unnestLonger_preservesRowOrder(self) -> None:
        df = pd.DataFrame({"g": [2, 1], "x": [[1, 2], [3]]})
        assert unnest_longer(df, "x")["g"].tolist() == [2, 2, 1]

    def test_unnestLonger_valuesTo(self) -> None:
        df = pd.DataFrame({"g": [1], "x": [[1, 2]]})
        assert unnest_longer(df, "x", values_to="val").columns.tolist() == ["g", "val"]

    def test_unnestLonger_indicesWithinEachList(self) -> None:
        df = pd.DataFrame({"g": [1, 2], "x": [[1, 2], [3]]})
        assert unnest_longer(df, "x", indices_include=True)["x_id"].tolist() == [1, 2, 1]

    # Unnest Wider
    def test_unnestWider_nonDefaultIndex(self) -> None:
        df = pd.DataFrame({"g": [1, 2], "x": [[1, 2], [3, 4]]}, index=[5, 6])
        actual = unnest_wider(df, "x")
        assert actual.index.tolist() == [5, 6]
        assert actual["x_2"].tolist() == [2, 4]

    def test_unnestWider_dictNamesSep(self) -> None:
        df = pd.DataFrame({"x": [{"a": 1}]})
        assert unnest_wider(df, "x", names_sep="_").columns.tolist() == ["x_a"]

    # Unnest
    def test_unnest_keepsRealNAs(self) -> None:
        df = pd.DataFrame({"g": [1, 2], "d": [{"x": [1, np.nan]}, {"x": [3]}]})
        actual = unnest(df, "d")
        assert actual["g"].tolist() == [1, 1, 2]
        assert actual.index.tolist() == [0, 1, 2]

    def test_unnest_dropsEmptyUnlessKeepEmpty(self) -> None:
        df = pd.DataFrame({"g": [1, 2], "d": [{"x": [1]}, {}]})
        assert unnest(df, "d")["g"].tolist() == [1]
        assert unnest(df, "d", keep_empty=True)["g"].tolist() == [1, 2]

    # Chop
    def test_chop_unsortedGroups(self) -> None:
        df = pd.DataFrame({"g": [2, 2, 1], "x": [1, 2, 3]})
        actual = chop(df, "x")
        assert actual["g"].tolist() == [2, 1]
        assert actual["x"].tolist() == [[1, 2], [3]]

    def test_chop_missingKey(self) -> None:
        df = pd.DataFrame({"g": [1.0, np.nan, 2.0], "x": [1, 2, 3]})
        assert chop(df, "x")["x"].tolist() == [[1], [2], [3]]

    # Unchop
    def test_unchop_keepsOtherNAs(self) -> None:
        df = pd.DataFrame({"g": [1, 2], "o": [np.nan, 1], "x": [[1, 2], [3]]})
        assert unchop(df, "x")["g"].tolist() == [1, 1, 2]

    def test_unchop_multipleColumns(self) -> None:
        df = pd.DataFrame({"g": [1, 2], "x": [[1, 2], [3]], "y": [["a", "b"], ["c"]]})
        actual = unchop(df, ["x", "y"])
        expected = pd.DataFrame({"g": [1, 1, 2], "x": [1, 2, 3], "y": ["a", "b", "c"]})
        pd.testing.assert_frame_equal(actual, expected)

    # Separate
    def test_separate_nonDefaultIndex(self) -> None:
        df = pd.DataFrame({"x": ["a_b", "c_d"]}, index=[5, 6])
        actual = separate(df, "x", ["A", "B"])
        expected = pd.DataFrame({"A": ["a", "c"], "B": ["b", "d"]}, index=[5, 6])
        pd.testing.assert_frame_equal(actual, expected)

    def test_separate_noFillWarningWhenNotNeeded(self, recwarn) -> None:
        separate(pd.DataFrame({"x": ["a_b", np.nan]}), "x", ["A", "B"])
        assert len(recwarn) == 0

    def test_separate_warningRowNumbers(self) -> None:
        df = pd.DataFrame({"x": ["a b", "a b c", "a"]})
        with pytest.warns(UserWarning) as record:
            separate(df, "x", ["A", "B"], " ")
        messages = [str(w.message) for w in record]
        assert any("discarded in 1 rows [2]" in m for m in messages)
        assert any("filled with `NA` in 1 rows [3]" in m for m in messages)

    def test_separate_tupleIntoAndPosition(self) -> None:
        df = pd.DataFrame({"a": [0], "x": ["1_2"], "z": [9]})
        actual = separate(df, "x", ("A", "B"), convert=True)
        assert actual.columns.tolist() == ["a", "A", "B", "z"]
        assert actual.iloc[0].tolist() == [0, 1, 2, 9]

    # Extract
    def test_extract_dropsNAColumn(self) -> None:
        df = pd.DataFrame({"x": ["a-b"]})
        actual = extract(df, "x", ["NA", "B"], "([a-z])-([a-z])")
        assert actual.columns.tolist() == ["B"]

    def test_extract_nonDefaultIndex(self) -> None:
        df = pd.DataFrame({"x": ["a-b", "c-d"]}, index=[5, 6])
        actual = extract(df, "x", ["A"])
        assert actual.index.tolist() == [5, 6]
        assert actual["A"].tolist() == ["a", "c"]

    # Unite
    def test_unite_leavesOtherColumnsAlone(self) -> None:
        df = pd.DataFrame({"x": ["a"], "y": ["NA"], "n": [np.nan]})
        actual = unite(df, "z", ["x", "y"])
        assert actual["z"].tolist() == ["a_NA"]
        assert actual["n"].dtype == np.float64

    def test_unite_naRmOtherSeparator(self) -> None:
        df = pd.DataFrame({"x": ["NA_ok", "x"], "y": [np.nan, np.nan]})
        actual = unite(df, "z", ["x", "y"], sep="-", na_rm=True)
        assert actual["z"].tolist() == ["NA_ok", "x"]

    def test_unite_numericColumns(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert unite(df, "z", ["a", "b"])["z"].tolist() == ["1_3", "2_4"]

    def test_unite_position(self) -> None:
        df = pd.DataFrame({"p": [0], "x": ["a"], "y": ["b"], "q": [1]})
        assert unite(df, "z", ["x", "y"]).columns.tolist() == ["p", "z", "q"]

    # Fill
    def test_fill_doesNotMutate(self) -> None:
        df = pd.DataFrame({"x": [1.0, np.nan]})
        fill(df, "x")
        assert np.isnan(df.loc[1, "x"])

    # Complete
    def test_complete_noColumns(self) -> None:
        df = pd.DataFrame({"x": [1]})
        pd.testing.assert_frame_equal(complete(df), df)

    def test_complete_nestingOnlyObservedPairs(self) -> None:
        df = pd.DataFrame({"g": [1, 2, 2], "id": [1, 2, 3], "name": ["a", "b", "c"]})
        actual = complete(df, ["g", "nesting(id, name)"])
        assert len(actual) == 6
        assert set(zip(actual["id"], actual["name"])) == {(1, "a"), (2, "b"), (3, "c")}


if __name__ == "__main__":
    pytest.main()
