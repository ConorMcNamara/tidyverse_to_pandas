from tidyverse.helper_select import everything, last_col
import pandas as pd
import numpy as np
import pytest


from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class TestHelperSelect:
    def test_everything(self) -> None:
        cars = pd.read_csv(DATA_DIR / "mtcars.csv")
        cars = cars.rename({"Unnamed: 0": "car"}, axis=1)
        expected = [
            "car",
            "mpg",
            "cyl",
            "disp",
            "hp",
            "drat",
            "wt",
            "qsec",
            "vs",
            "am",
            "gear",
            "carb",
        ]
        np.testing.assert_array_equal(everything(cars), expected)

    def test_lastCol_noOffset(self) -> None:
        cars = pd.read_csv(DATA_DIR / "mtcars.csv")
        cars = cars.rename({"Unnamed: 0": "car"}, axis=1)
        assert last_col(cars) == "carb"

    def test_lastCol_offset(self) -> None:
        cars = pd.read_csv(DATA_DIR / "mtcars.csv")
        cars = cars.rename({"Unnamed: 0": "car"}, axis=1)
        assert last_col(cars, 2) == "am"


if __name__ == "__main__":
    pytest.main()
