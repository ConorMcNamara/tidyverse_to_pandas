import numpy as np
import pandas as pd
import pytest

from tidyverse_to_pandas import dplyr_to_pandas as dplyr


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})


class TestLegacyMutate:
    def test_no_whitespace_around_equals(self, df) -> None:
        assert dplyr.mutate(df, "c=a+b")["c"].tolist() == [11, 22, 33, 44]

    def test_comparison_in_expression(self, df) -> None:
        assert dplyr.mutate(df, "c = a == 2")["c"].tolist() == [False, True, False, False]

    def test_numeric_literal(self, df) -> None:
        assert dplyr.mutate(df, "c = 2 * a")["c"].tolist() == [2, 4, 6, 8]

    def test_transmute(self, df) -> None:
        result = dplyr.transmute(df, "c = a + b")
        assert result.columns.tolist() == ["c"]


class TestLegacySelect:
    def test_num_range_multi_digit(self) -> None:
        data = pd.DataFrame({f"x{i}": [i] for i in range(1, 12)})
        assert dplyr.select(data, "num_range(x, 1:10)").columns.tolist() == [f"x{i}" for i in range(1, 11)]

    def test_drop_then_keep_has_no_duplicates(self, df) -> None:
        assert dplyr.select(df, "-a", "b").columns.tolist() == ["b"]

    def test_quoted_helper(self) -> None:
        data = pd.DataFrame({"description": [1], "other": [2]})
        assert dplyr.select(data, 'starts_with("des")').columns.tolist() == ["description"]

    def test_last_col(self, df) -> None:
        assert dplyr.select(df, "last_col()").columns.tolist() == ["b"]
        assert dplyr.select(df, "last_col(offset = 1)").columns.tolist() == ["a"]


class TestLegacySummarise:
    def test_mad(self) -> None:
        data = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
        assert dplyr.summarise(data, "m = mad(a)")["m"].iloc[0] == pytest.approx(1.4826)

    def test_first_last_after_filter(self, df) -> None:
        filtered = df[df["a"] > 1]
        result = dplyr.summarise(filtered, "f = first(a)", "l = last(a)")
        assert result.iloc[0].tolist() == [2, 4]

    def test_sum_and_unknown_function(self, df) -> None:
        assert dplyr.summarise(df, "s = sum(a)")["s"].iloc[0] == 10
        with pytest.raises(ValueError):
            dplyr.summarise(df, "s = not_a_function(a)")

    def test_unnamed_expression(self, df) -> None:
        assert dplyr.summarise(df, "mean(a)")["mean(a)"].iloc[0] == 2.5


class TestLegacyFilterArrange:
    def test_filter_precedence(self, df) -> None:
        assert dplyr.filter(df, "a == 1 | a == 4", "b > 15")["a"].tolist() == [4]

    def test_filter_aggregate(self, df) -> None:
        assert dplyr.filter(df, "b > mean(b) * 1.5")["b"].tolist() == [40]

    def test_arrange_desc_column_name(self) -> None:
        data = pd.DataFrame({"description": [2, 1, np.nan]})
        assert dplyr.arrange(data, "desc(description)")["description"].tolist()[:2] == [2, 1]

    def test_rename_without_spaces(self, df) -> None:
        assert dplyr.rename(df, "a=x").columns.tolist() == ["x", "b"]
