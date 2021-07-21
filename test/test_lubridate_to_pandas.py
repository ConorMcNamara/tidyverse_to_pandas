import pytest

import numpy as np
import pandas as pd

from context import lubridate_to_pandas as ltp


class TestYMD:

    @staticmethod
    def test_ymd_typeError():
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.ymd(date)

    @staticmethod
    def test_ymd_string():
        date = '20/02/01'
        expected_date = '2020-02-01'
        assert ltp.ymd(date) == expected_date

        timezone = 'US/Pacific'
        expected_date = '2020-02-01 PST'
        assert ltp.ymd(date, timezone) == expected_date

    @staticmethod
    def test_ymd_list():
        date = ['20-02-01']
        expected_date = ['2020-02-01']
        np.testing.assert_array_equal(expected_date, ltp.ymd(date))

        timezone = 'US/Eastern'
        expected_date = ['2020-02-01 EST']
        np.testing.assert_array_equal(expected_date, ltp.ymd(date, timezone))

    @staticmethod
    def test_ymd_numpy():
        date = np.array(['20 Jan 1'])
        expected_date = np.array(['2020-01-01'], dtype='datetime64[D]')
        np.testing.assert_array_equal(expected_date, ltp.ymd(date))

        timezone = 'US/Mountain'
        expected_date = np.array(['2020-01-01T07:00:00.000000000'], dtype='datetime64[ns]')
        np.testing.assert_array_equal(expected_date, ltp.ymd(date, timezone))

    @staticmethod
    def test_ymd_pandas():
        date = pd.Series(['20 02 01'])
        expected_date = pd.Series(['2020-02-01'], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(expected_date, ltp.ymd(date))

        timezone = 'US/Central'
        expected_date = pd.Series(['2020-02-01 00:00:00-06:00'], dtype='datetime64[ns, US/Central]')
        pd.testing.assert_series_equal(expected_date, ltp.ymd(date, timezone))


class TestYDM:

    @staticmethod
    def test_ydm_typeError():
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.ydm(date)

    @staticmethod
    def test_ydm_string():
        date = '20/02/01'
        expected_date = '2020-01-02'
        assert ltp.ydm(date) == expected_date

        timezone = 'US/Pacific'
        expected_date = '2020-01-02 PST'
        assert ltp.ydm(date, timezone) == expected_date

    @staticmethod
    def test_ymd_list():
        date = ['20-02-01']
        expected_date = ['2020-01-02']
        np.testing.assert_array_equal(expected_date, ltp.ydm(date))

        timezone = 'US/Eastern'
        expected_date = ['2020-01-02 EST']
        np.testing.assert_array_equal(expected_date, ltp.ydm(date, timezone))

    @staticmethod
    def test_ymd_numpy():
        date = np.array(['20 Jan 1'])
        expected_date = np.array(['2020-01-01'], dtype='datetime64[D]')
        np.testing.assert_array_equal(expected_date, ltp.ydm(date))

        timezone = 'US/Mountain'
        expected_date = np.array(['2020-01-01T07:00:00.000000000'], dtype='datetime64[ns]')
        np.testing.assert_array_equal(expected_date, ltp.ydm(date, timezone))

    @staticmethod
    def test_ymd_pandas():
        date = pd.Series(['20 02 01'])
        expected_date = pd.Series(['2020-01-02'], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(expected_date, ltp.ydm(date))

        timezone = 'US/Central'
        expected_date = pd.Series(['2020-01-02 00:00:00-06:00'], dtype='datetime64[ns, US/Central]')
        pd.testing.assert_series_equal(expected_date, ltp.ydm(date, timezone))


class TestMDY:

    @staticmethod
    def test_mdy_typeError():
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.mdy(date)

    @staticmethod
    def test_mdy_string():
        date = '10/02/01'
        expected_date = '2001-10-02'
        assert ltp.mdy(date) == expected_date

        timezone = 'US/Pacific'
        expected_date = '2001-10-02 PDT'
        assert ltp.mdy(date, timezone) == expected_date

    @staticmethod
    def test_mdy_list():
        date = ['10-02-01']
        expected_date = ['2001-10-02']
        np.testing.assert_array_equal(expected_date, ltp.mdy(date))

        timezone = 'US/Eastern'
        expected_date = ['2001-10-02 EDT']
        np.testing.assert_array_equal(expected_date, ltp.mdy(date, timezone))

    @staticmethod
    def test_mdy_numpy():
        date = np.array(['20 1 Jan'])
        expected_date = np.array(['2020-01-01'], dtype='datetime64[D]')
        np.testing.assert_array_equal(expected_date, ltp.mdy(date))

        timezone = 'US/Mountain'
        expected_date = np.array(['2020-01-01T07:00:00.000000000'], dtype='datetime64[ns]')
        np.testing.assert_array_equal(expected_date, ltp.mdy(date, timezone))

    @staticmethod
    def test_mdy_pandas():
        date = pd.Series(['10 02 01'])
        expected_date = pd.Series(['2001-10-02'], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(expected_date, ltp.mdy(date))

        timezone = 'US/Central'
        expected_date = pd.Series(['2001-10-02 00:00:00-05:00'], dtype='datetime64[ns, US/Central]')
        pd.testing.assert_series_equal(expected_date, ltp.mdy(date, timezone))


class TestMYD:

    @staticmethod
    def test_myd_typeError():
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.myd(date)

    @staticmethod
    def test_myd_string():
        date = '10/02/01'
        expected_date = '2002-10-01'
        assert ltp.myd(date) == expected_date

        timezone = 'US/Pacific'
        expected_date = '2002-10-01 PDT'
        assert ltp.myd(date, timezone) == expected_date

    @staticmethod
    def test_myd_list():
        date = ['10-02-01']
        expected_date = ['2002-10-01']
        np.testing.assert_array_equal(expected_date, ltp.myd(date))

        timezone = 'US/Eastern'
        expected_date = ['2002-10-01 EDT']
        np.testing.assert_array_equal(expected_date, ltp.myd(date, timezone))

    @staticmethod
    def test_myd_numpy():
        date = np.array(['Jan 20 01'])
        expected_date = np.array(['2020-01-01'], dtype='datetime64[D]')
        np.testing.assert_array_equal(expected_date, ltp.myd(date))

        timezone = 'US/Mountain'
        expected_date = np.array(['2020-01-01T07:00:00.000000000'], dtype='datetime64[ns]')
        np.testing.assert_array_equal(expected_date, ltp.myd(date, timezone))

    @staticmethod
    def test_myd_pandas():
        date = pd.Series(['10 02 01'])
        expected_date = pd.Series(['2002-10-01'], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(expected_date, ltp.myd(date))

        timezone = 'US/Central'
        expected_date = pd.Series(['2002-10-01 00:00:00-05:00'], dtype='datetime64[ns, US/Central]')
        pd.testing.assert_series_equal(expected_date, ltp.myd(date, timezone))


class TestDMY:
    @staticmethod
    def test_dmy_typeError():
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.dmy(date)

    @staticmethod
    def test_dmy_string():
        date = '10/02/01'
        expected_date = '2001-02-10'
        assert ltp.dmy(date) == expected_date

        timezone = 'US/Pacific'
        expected_date = '2001-02-10 PST'
        assert ltp.dmy(date, timezone) == expected_date

    @staticmethod
    def test_dmy_list():
        date = ['10-02-01']
        expected_date = ['2001-02-10']
        np.testing.assert_array_equal(expected_date, ltp.dmy(date))

        timezone = 'US/Eastern'
        expected_date = ['2001-02-10 EST']
        np.testing.assert_array_equal(expected_date, ltp.dmy(date, timezone))

    @staticmethod
    def test_dmy_numpy():
        date = np.array(['01 Jan 20'])
        expected_date = np.array(['2020-01-01'], dtype='datetime64[D]')
        np.testing.assert_array_equal(expected_date, ltp.dmy(date))

        timezone = 'US/Mountain'
        expected_date = np.array(['2020-01-01T07:00:00.000000000'], dtype='datetime64[ns]')
        np.testing.assert_array_equal(expected_date, ltp.dmy(date, timezone))

    @staticmethod
    def test_dmy_pandas():
        date = pd.Series(['10 02 01'])
        expected_date = pd.Series(['2001-02-10'], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(expected_date, ltp.dmy(date))

        timezone = 'US/Central'
        expected_date = pd.Series(['2001-02-10 00:00:00-06:00'], dtype='datetime64[ns, US/Central]')
        pd.testing.assert_series_equal(expected_date, ltp.dmy(date, timezone))


class TestDYM:
    @staticmethod
    def test_dym_typeError():
        date = 201001
        with pytest.raises(TypeError, match="Cannot identify date variable"):
            ltp.dym(date)

    @staticmethod
    def test_dym_string():
        date = '10/02/01'
        expected_date = '2002-01-10'
        assert ltp.dym(date) == expected_date

        timezone = 'US/Pacific'
        expected_date = '2002-01-10 PST'
        assert ltp.dym(date, timezone) == expected_date

    @staticmethod
    def test_dym_list():
        date = ['10-02-01']
        expected_date = ['2002-01-10']
        np.testing.assert_array_equal(expected_date, ltp.dym(date))

        timezone = 'US/Eastern'
        expected_date = ['2002-01-10 EST']
        np.testing.assert_array_equal(expected_date, ltp.dym(date, timezone))

    @staticmethod
    def test_dym_numpy():
        date = np.array(['01 20 Jan'])
        expected_date = np.array(['2020-01-01'], dtype='datetime64[D]')
        np.testing.assert_array_equal(expected_date, ltp.dym(date))

        timezone = 'US/Mountain'
        expected_date = np.array(['2020-01-01T07:00:00.000000000'], dtype='datetime64[ns]')
        np.testing.assert_array_equal(expected_date, ltp.dym(date, timezone))

    @staticmethod
    def test_dym_pandas():
        date = pd.Series(['10 02 01'])
        expected_date = pd.Series(['2002-01-10'], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(expected_date, ltp.dym(date))

        timezone = 'US/Central'
        expected_date = pd.Series(['2002-01-10 00:00:00-06:00'], dtype='datetime64[ns, US/Central]')
        pd.testing.assert_series_equal(expected_date, ltp.dym(date, timezone))