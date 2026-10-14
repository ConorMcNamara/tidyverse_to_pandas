import numpy as np
import pandas as pd
import pytest

import tidyverse_to_pandas.lubridate_to_pandas as ltp


class TestYMD:
    @staticmethod
    def test_ymd_typeError() -> None:
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.ymd(date)

    @staticmethod
    def test_ymd_string() -> None:
        date = "20/02/01"
        expected_date = "2020-02-01"
        assert ltp.ymd(date) == expected_date

        timezone = "US/Pacific"
        expected_date = "2020-02-01 PST"
        assert ltp.ymd(date, timezone) == expected_date

    @staticmethod
    def test_ymd_list() -> None:
        date = ["20-02-01"]
        expected_date = ["2020-02-01"]
        np.testing.assert_array_equal(expected_date, ltp.ymd(date))

        timezone = "US/Eastern"
        expected_date = ["2020-02-01 EST"]
        np.testing.assert_array_equal(expected_date, ltp.ymd(date, timezone))

    @staticmethod
    def test_ymd_numpy() -> None:
        date = np.array(["20 Jan 1"])
        expected_date = np.array(["2020-01-01"], dtype="datetime64[D]")
        np.testing.assert_array_equal(expected_date, ltp.ymd(date))

        timezone = "US/Mountain"
        expected_date = np.array(["2020-01-01T07:00:00.000000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(expected_date, ltp.ymd(date, timezone))

    @staticmethod
    def test_ymd_pandas() -> None:
        date = pd.Series(["20 02 01"])
        expected_date = pd.Series(["2020-02-01"], dtype="datetime64[ns]")
        pd.testing.assert_series_equal(expected_date, ltp.ymd(date))

        timezone = "US/Central"
        expected_date = pd.Series(["2020-02-01 00:00:00-06:00"], dtype="datetime64[ns, US/Central]")
        pd.testing.assert_series_equal(expected_date, ltp.ymd(date, timezone))


class TestYDM:
    @staticmethod
    def test_ydm_typeError() -> None:
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.ydm(date)

    @staticmethod
    def test_ydm_string() -> None:
        date = "20/02/01"
        expected_date = "2020-01-02"
        assert ltp.ydm(date) == expected_date

        timezone = "US/Pacific"
        expected_date = "2020-01-02 PST"
        assert ltp.ydm(date, timezone) == expected_date

    @staticmethod
    def test_ydm_list() -> None:
        date = ["20-02-01"]
        expected_date = ["2020-01-02"]
        np.testing.assert_array_equal(expected_date, ltp.ydm(date))

        timezone = "US/Eastern"
        expected_date = ["2020-01-02 EST"]
        np.testing.assert_array_equal(expected_date, ltp.ydm(date, timezone))

    @staticmethod
    def test_ydm_numpy() -> None:
        date = np.array(["20 Jan 1"])
        expected_date = np.array(["2020-01-01"], dtype="datetime64[D]")
        np.testing.assert_array_equal(expected_date, ltp.ydm(date))

        timezone = "US/Mountain"
        expected_date = np.array(["2020-01-01T07:00:00.000000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(expected_date, ltp.ydm(date, timezone))

    @staticmethod
    def test_ydm_pandas() -> None:
        date = pd.Series(["20 02 01"])
        expected_date = pd.Series(["2020-01-02"], dtype="datetime64[ns]")
        pd.testing.assert_series_equal(expected_date, ltp.ydm(date))

        timezone = "US/Central"
        expected_date = pd.Series(["2020-01-02 00:00:00-06:00"], dtype="datetime64[ns, US/Central]")
        pd.testing.assert_series_equal(expected_date, ltp.ydm(date, timezone))


class TestMDY:
    @staticmethod
    def test_mdy_typeError() -> None:
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.mdy(date)

    @staticmethod
    def test_mdy_string() -> None:
        date = "10/02/01"
        expected_date = "2001-10-02"
        assert ltp.mdy(date) == expected_date

        timezone = "US/Pacific"
        expected_date = "2001-10-02 PDT"
        assert ltp.mdy(date, timezone) == expected_date

    @staticmethod
    def test_mdy_list() -> None:
        date = ["10-02-01"]
        expected_date = ["2001-10-02"]
        np.testing.assert_array_equal(expected_date, ltp.mdy(date))

        timezone = "US/Eastern"
        expected_date = ["2001-10-02 EDT"]
        np.testing.assert_array_equal(expected_date, ltp.mdy(date, timezone))

    @staticmethod
    def test_mdy_numpy() -> None:
        # Numbers fill the remaining fields in mdy order: day 20, year (20)01
        date = np.array(["20 1 Jan"])
        expected_date = np.array(["2001-01-20"], dtype="datetime64[D]")
        np.testing.assert_array_equal(expected_date, ltp.mdy(date))

        timezone = "US/Mountain"
        expected_date = np.array(["2001-01-20T07:00:00.000000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(expected_date, ltp.mdy(date, timezone))

    @staticmethod
    def test_mdy_pandas() -> None:
        date = pd.Series(["10 02 01"])
        expected_date = pd.Series(["2001-10-02"], dtype="datetime64[ns]")
        pd.testing.assert_series_equal(expected_date, ltp.mdy(date))

        timezone = "US/Central"
        expected_date = pd.Series(["2001-10-02 00:00:00-05:00"], dtype="datetime64[ns, US/Central]")
        pd.testing.assert_series_equal(expected_date, ltp.mdy(date, timezone))


class TestMYD:
    @staticmethod
    def test_myd_typeError() -> None:
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.myd(date)

    @staticmethod
    def test_myd_string() -> None:
        date = "10/02/01"
        expected_date = "2002-10-01"
        assert ltp.myd(date) == expected_date

        timezone = "US/Pacific"
        expected_date = "2002-10-01 PDT"
        assert ltp.myd(date, timezone) == expected_date

    @staticmethod
    def test_myd_list() -> None:
        date = ["10-02-01"]
        expected_date = ["2002-10-01"]
        np.testing.assert_array_equal(expected_date, ltp.myd(date))

        timezone = "US/Eastern"
        expected_date = ["2002-10-01 EDT"]
        np.testing.assert_array_equal(expected_date, ltp.myd(date, timezone))

    @staticmethod
    def test_myd_numpy() -> None:
        date = np.array(["Jan 20 01"])
        expected_date = np.array(["2020-01-01"], dtype="datetime64[D]")
        np.testing.assert_array_equal(expected_date, ltp.myd(date))

        timezone = "US/Mountain"
        expected_date = np.array(["2020-01-01T07:00:00.000000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(expected_date, ltp.myd(date, timezone))

    @staticmethod
    def test_myd_pandas() -> None:
        date = pd.Series(["10 02 01"])
        expected_date = pd.Series(["2002-10-01"], dtype="datetime64[ns]")
        pd.testing.assert_series_equal(expected_date, ltp.myd(date))

        timezone = "US/Central"
        expected_date = pd.Series(["2002-10-01 00:00:00-05:00"], dtype="datetime64[ns, US/Central]")
        pd.testing.assert_series_equal(expected_date, ltp.myd(date, timezone))


class TestDMY:
    @staticmethod
    def test_dmy_typeError() -> None:
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.dmy(date)

    @staticmethod
    def test_dmy_string() -> None:
        date = "10/02/01"
        expected_date = "2001-02-10"
        assert ltp.dmy(date) == expected_date

        timezone = "US/Pacific"
        expected_date = "2001-02-10 PST"
        assert ltp.dmy(date, timezone) == expected_date

    @staticmethod
    def test_dmy_list() -> None:
        date = ["10-02-01"]
        expected_date = ["2001-02-10"]
        np.testing.assert_array_equal(expected_date, ltp.dmy(date))

        timezone = "US/Eastern"
        expected_date = ["2001-02-10 EST"]
        np.testing.assert_array_equal(expected_date, ltp.dmy(date, timezone))

    @staticmethod
    def test_dmy_numpy() -> None:
        date = np.array(["01 Jan 20"])
        expected_date = np.array(["2020-01-01"], dtype="datetime64[D]")
        np.testing.assert_array_equal(expected_date, ltp.dmy(date))

        timezone = "US/Mountain"
        expected_date = np.array(["2020-01-01T07:00:00.000000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(expected_date, ltp.dmy(date, timezone))

    @staticmethod
    def test_dmy_pandas() -> None:
        date = pd.Series(["10 02 01"])
        expected_date = pd.Series(["2001-02-10"], dtype="datetime64[ns]")
        pd.testing.assert_series_equal(expected_date, ltp.dmy(date))

        timezone = "US/Central"
        expected_date = pd.Series(["2001-02-10 00:00:00-06:00"], dtype="datetime64[ns, US/Central]")
        pd.testing.assert_series_equal(expected_date, ltp.dmy(date, timezone))


class TestDYM:
    @staticmethod
    def test_dym_typeError() -> None:
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.dym(date)

    @staticmethod
    def test_dym_string() -> None:
        date = "10/02/01"
        expected_date = "2002-01-10"
        assert ltp.dym(date) == expected_date

        timezone = "US/Pacific"
        expected_date = "2002-01-10 PST"
        assert ltp.dym(date, timezone) == expected_date

    @staticmethod
    def test_dym_list() -> None:
        date = ["10-02-01"]
        expected_date = ["2002-01-10"]
        np.testing.assert_array_equal(expected_date, ltp.dym(date))

        timezone = "US/Eastern"
        expected_date = ["2002-01-10 EST"]
        np.testing.assert_array_equal(expected_date, ltp.dym(date, timezone))

    @staticmethod
    def test_dym_numpy() -> None:
        date = np.array(["01 20 Jan"])
        expected_date = np.array(["2020-01-01"], dtype="datetime64[D]")
        np.testing.assert_array_equal(expected_date, ltp.dym(date))

        timezone = "US/Mountain"
        expected_date = np.array(["2020-01-01T07:00:00.000000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(expected_date, ltp.dym(date, timezone))

    @staticmethod
    def test_dym_pandas() -> None:
        date = pd.Series(["10 02 01"])
        expected_date = pd.Series(["2002-01-10"], dtype="datetime64[ns]")
        pd.testing.assert_series_equal(expected_date, ltp.dym(date))

        timezone = "US/Central"
        expected_date = pd.Series(["2002-01-10 00:00:00-06:00"], dtype="datetime64[ns, US/Central]")
        pd.testing.assert_series_equal(expected_date, ltp.dym(date, timezone))


if __name__ == "__main__":
    pytest.main()


class TestParsingRegressions:
    @staticmethod
    def test_two_digit_year_cutoff() -> None:
        assert ltp.myd("01/99/02") == "1999-01-02"
        assert ltp.dym("02/99/01") == "1999-01-02"
        assert ltp.ymd("68-01-01") == "2068-01-01"
        assert ltp.ymd("69-01-01") == "1969-01-01"

    @staticmethod
    def test_compact_four_digits() -> None:
        assert ltp.myd("1209") == "2020-01-09"
        assert ltp.dym("9201") == "2020-01-09"

    @staticmethod
    def test_compact_eight_digits() -> None:
        assert ltp.ymd("20200102") == "2020-01-02"
        assert ltp.dmy("02012020") == "2020-01-02"

    @staticmethod
    def test_month_names() -> None:
        assert ltp.mdy("January 5th, 2024") == "2024-01-05"
        assert ltp.dmy("5 Sept 2024") == "2024-09-05"

    @staticmethod
    def test_numpy_string_scalar() -> None:
        assert ltp.ymd(np.str_("2020-01-02")) == "2020-01-02"

    @staticmethod
    def test_invalid_dates_are_missing() -> None:
        with pytest.warns(UserWarning, match="1 failed to parse"):
            assert ltp.mdy("13/01/2020") is None
        with pytest.warns(UserWarning, match="1 failed to parse"):
            result = ltp.ymd(pd.Series(["2020-01-02", "2020-02-30"], index=[5, 6]))
        assert result.index.tolist() == [5, 6]
        assert pd.isna(result[6])

    @staticmethod
    def test_missing_input_stays_missing() -> None:
        assert ltp.ymd(["2020-01-02", None]) == ["2020-01-02", None]


class TestYMDHMS:
    @staticmethod
    def test_ymd_hms_string() -> None:
        assert ltp.ymd_hms("2020-01-02 03:04:05") == "2020-01-02 03:04:05"
        assert ltp.ymd_hms("2020-01-02T03:04:05", tz="UTC") == "2020-01-02 03:04:05 UTC"

    @staticmethod
    def test_ymd_hms_pandas() -> None:
        result = ltp.ymd_hms(pd.Series(["2020-01-02 03:04:05.5"]))
        expected = pd.Series(pd.to_datetime(["2020-01-02 03:04:05.5"])).astype("datetime64[ns]")
        pd.testing.assert_series_equal(result, expected)

    @staticmethod
    def test_ymd_hms_typeError() -> None:
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.ymd_hms(201001)
