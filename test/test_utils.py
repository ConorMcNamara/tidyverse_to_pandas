import pandas as pd
import pytest

from tidyverse_to_pandas import utils


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1], "b": [2], "c": [3], "first-name": [4]})


class TestColumnSelection:
    def test_multiple_exclusions(self, df):
        assert utils._get_list_columns(df, ["-a", "-b"]) == ["c", "first-name"]

    def test_exclusion_after_inclusion(self, df):
        assert utils._get_list_columns(df, ["a", "b", "-a"]) == ["b"]

    def test_hyphenated_name(self, df):
        assert utils._get_str_columns(df, "first-name") == ["first-name"]
        assert utils._get_str_columns(df, "-first-name") == ["a", "b", "c"]

    def test_range(self, df):
        assert utils._get_str_columns(df, "b:first-name") == ["b", "c", "first-name"]


class TestConvertNumeric:
    def test_signed_and_decimal(self):
        out = utils._convert_numeric(pd.DataFrame({"x": ["1.5", "-2"], "y": [1, 2], "z": ["a", "1"]}))
        assert out["x"].tolist() == [1.5, -2.0]
        assert out["y"].tolist() == [1, 2]
        assert out["z"].tolist() == ["a", "1"]


class TestCheckUnique:
    def test_unique_avoids_existing_names(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "a.1"])
        assert list(utils._check_unique(df).columns) == ["a", "a.2", "a.1"]
