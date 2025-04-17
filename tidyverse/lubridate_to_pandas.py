import dateutil
import re

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from pytz import timezone
from datetime import date, datetime


# Year, month and day parsing
def ymd(
    dates: Union[str, Sequence[str], np.ndarray, pd.Series], tz: Optional[str] = None
) -> Union[str, Sequence[str], np.ndarray, pd.Series]:
    """Converts our suspected dates in ymd format to %Y-%m-%d

    Parameters
    ----------
    dates: str or array-like format
        A character or an array-like of suspected dates
    tz: str
        Time zone indicator. If None (default), a Date object is returned. Otherwise a POSIXct with time zone attribute set to tz.

    Returns
    -------
    Our suspected dates converted to %Y-%m-%d format
    """
    if tz is not None:
        zone = timezone(tz)
    if isinstance(dates, pd.Series):
        return_date = pd.to_datetime(dates, yearfirst=True, dayfirst=False)
        if tz is not None:
            return_date = return_date.dt.tz_localize(tz=zone)
    elif isinstance(dates, (np.ndarray, np.generic)):
        if tz is None:
            return_date = np.array(
                [dateutil.parser.parse(date, yearfirst=True, dayfirst=False).strftime("%Y-%m-%d") for date in dates],
                dtype="datetime64",
            )
        else:
            return_date = (
                pd.to_datetime(pd.Series(dates), yearfirst=True, dayfirst=False)
                .dt.tz_localize(tz=zone)
                .to_numpy(dtype="datetime64")
            )
    elif isinstance(dates, (list, tuple)):
        if tz is None:
            return_date = [
                dateutil.parser.parse(date, yearfirst=True, dayfirst=False).strftime("%Y-%m-%d") for date in dates
            ]
        else:
            return_date = [
                zone.localize(dateutil.parser.parse(date, yearfirst=True, dayfirst=False)).strftime("%Y-%m-%d %Z")
                for date in dates
            ]
    elif isinstance(dates, str):
        if tz is None:
            return_date = dateutil.parser.parse(dates, yearfirst=True, dayfirst=False).strftime("%Y-%m-%d")
        else:
            return_date = zone.localize(dateutil.parser.parse(dates, yearfirst=True, dayfirst=False)).strftime(
                "%Y-%m-%d %Z"
            )
    else:
        raise TypeError("Cannot identify date variable")
    return return_date


def ydm(
    dates: Union[str, Sequence[str], np.ndarray, pd.Series], tz: Optional[str] = None
) -> Union[str, Sequence[str], np.ndarray, pd.Series]:
    """Converts our suspected dates in ydm format to %Y-%m-%d

    Parameters
    ----------
    dates: str or array-like format
        A character or an array-like of suspected dates
    tz: str
        Time zone indicator. If None (default), a Date object is returned. Otherwise a POSIXct with time zone attribute set to tz.

    Returns
    -------
    Our suspected dates converted to %Y-%d-%m format
    """
    if tz is not None:
        zone = timezone(tz)
    if isinstance(dates, pd.Series):
        return_date = pd.to_datetime(dates, dayfirst=True, yearfirst=True)
        if tz is not None:
            return_date = return_date.dt.tz_localize(tz=zone)
    elif isinstance(dates, (np.ndarray, np.generic)):
        if tz is None:
            return_date = np.array(
                [dateutil.parser.parse(date, dayfirst=True, yearfirst=True).strftime("%Y-%m-%d") for date in dates],
                dtype="datetime64",
            )
        else:
            return_date = (
                pd.to_datetime(pd.Series(dates), dayfirst=True, yearfirst=True)
                .dt.tz_localize(tz=zone)
                .to_numpy(dtype="datetime64")
            )
    elif isinstance(dates, (list, tuple)):
        if tz is None:
            return_date = [
                dateutil.parser.parse(date, dayfirst=True, yearfirst=True).strftime("%Y-%m-%d") for date in dates
            ]
        else:
            return_date = [
                zone.localize(dateutil.parser.parse(date, dayfirst=True, yearfirst=True)).strftime("%Y-%m-%d %Z")
                for date in dates
            ]
    elif isinstance(dates, str):
        if tz is None:
            return_date = dateutil.parser.parse(dates, dayfirst=True, yearfirst=True).strftime("%Y-%m-%d")
        else:
            return_date = zone.localize(dateutil.parser.parse(dates, dayfirst=True, yearfirst=True)).strftime(
                "%Y-%m-%d %Z"
            )
    else:
        raise TypeError("Cannot identify date variable")
    return return_date


def mdy(
    dates: Union[str, Sequence[str], np.ndarray, pd.Series], tz: Optional[str] = None
) -> Union[str, Sequence[str], np.ndarray, pd.Series]:
    """Converts our suspected dates in mdy format to %Y-%m-%d

    Parameters
    ----------
    dates: str or array-like format
        A character or an array-like of suspected dates
    tz: str
        Time zone indicator. If None (default), a Date object is returned. Otherwise a POSIXct with time zone attribute set to tz.

    Returns
    -------
    Our suspected dates converted to %Y-%d-%m format
    """
    if tz is not None:
        zone = timezone(tz)
    if isinstance(dates, pd.Series):
        return_date = pd.to_datetime(dates, dayfirst=False, yearfirst=False)
        if tz is not None:
            return_date = return_date.dt.tz_localize(tz=zone)
    elif isinstance(dates, (np.ndarray, np.generic)):
        if tz is None:
            return_date = np.array(
                [dateutil.parser.parse(date, dayfirst=False, yearfirst=False).strftime("%Y-%m-%d") for date in dates],
                dtype="datetime64",
            )
        else:
            return_date = (
                pd.to_datetime(pd.Series(dates), dayfirst=False, yearfirst=False)
                .dt.tz_localize(tz=zone)
                .to_numpy(dtype="datetime64")
            )
    elif isinstance(dates, (list, tuple)):
        if tz is None:
            return_date = [
                dateutil.parser.parse(date, dayfirst=False, yearfirst=False).strftime("%Y-%m-%d") for date in dates
            ]
        else:
            return_date = [
                zone.localize(dateutil.parser.parse(date, dayfirst=False, yearfirst=False)).strftime("%Y-%m-%d %Z")
                for date in dates
            ]
    elif isinstance(dates, str):
        if tz is None:
            return_date = dateutil.parser.parse(dates, dayfirst=False, yearfirst=False).strftime("%Y-%m-%d")
        else:
            return_date = zone.localize(dateutil.parser.parse(dates, dayfirst=False, yearfirst=False)).strftime(
                "%Y-%m-%d %Z"
            )
    else:
        raise TypeError("Cannot identify date variable")
    return return_date


def _myd(dates):
    current_year = date.today().year
    month_cal = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    # Cover when the month is in alphabetical format
    if dates[0].isalpha():
        month = month_cal[dates[0:3].capitalize()]
        all_numerics = re.findall(r"\d+", dates)
        if len(all_numerics) == 2:
            year = all_numerics[0]
            day = all_numerics[1]
        elif len(all_numerics) == 1:
            if len(all_numerics[0]) in [5, 6]:
                year = all_numerics[0][0:4]
                day = all_numerics[0][4:]
            elif len(all_numerics[0]) in [3, 4]:
                year = all_numerics[0][0:2]
                day = all_numerics[0][2:]
            else:
                raise ValueError("All formats failed to parse. No formats found.")
        else:
            raise ValueError("All formats failed to parse. No formats found.")
    # Cover when the month is in numerical format
    else:
        if "/" in dates:
            date_split = dates.split("/")
        elif "-" in dates:
            date_split = dates.split("-")
        elif "." in dates:
            date_split = dates.split(".")
        elif " " in dates:
            date_split = dates.split(" ")
        else:
            if len(dates) == 4:
                date_split = [dates[1], dates[1:3], dates[3]]
            elif len(dates) == 6:
                date_split = [dates[0:2], dates[2:4], dates[4:]]
            elif len(dates) == 8:
                date_split = [dates[0:2], dates[2:6], dates[6:]]
            else:
                raise ValueError("All formats failed to parse. No formats found.")
        month, year, day = date_split[0], date_split[1], date_split[2]
    if len(year) == 2:
        if int(year) > current_year:
            year = "19" + year
        else:
            year = "20" + year
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    return datetime.strptime("{}-{}-{}".format(year, month, day), "%Y-%m-%d")


def myd(dates, tz=None):
    """Converts our suspected dates in myd format to %Y-%m-%d

    Parameters
    ----------
    dates: str or array-like format
        A character or an array-like of suspected dates
    tz: str
        Time zone indicator. If None (default), a Date object is returned. Otherwise a POSIXct with time zone attribute set to tz.

    Returns
    -------
    Our suspected dates converted to %Y-%d-%m format
    """
    if tz is not None:
        zone = timezone(tz)
    if isinstance(dates, pd.Series):
        return_date = pd.to_datetime(dates.apply(lambda x: _myd(x).strftime("%Y-%m-%d")))
        if tz is not None:
            return_date = return_date.dt.tz_localize(tz=zone)
    elif isinstance(dates, (np.ndarray, np.generic)):
        if tz is None:
            return_date = np.array(list(map(_myd, dates)), dtype="datetime64[D]")
        else:
            return_date = (
                pd.to_datetime(pd.Series(dates).apply(lambda x: _myd(x).strftime("%Y-%m-%d")))
                .dt.tz_localize(tz=zone)
                .to_numpy(dtype="datetime64")
            )
    elif isinstance(dates, (list, tuple)):
        if tz is None:
            return_date = [_myd(date).strftime("%Y-%m-%d") for date in dates]
        else:
            return_date = [zone.localize(_myd(date)).strftime("%Y-%m-%d %Z") for date in dates]
    elif isinstance(dates, str):
        if tz is None:
            return_date = _myd(dates).strftime("%Y-%m-%d")
        else:
            return_date = zone.localize(_myd(dates)).strftime("%Y-%m-%d %Z")
    else:
        raise TypeError("Cannot identify date variable")
    return return_date


def dmy(
    dates: Union[str, Sequence[str], np.ndarray, pd.Series], tz: Optional[str] = None
) -> Union[str, Sequence[str], np.ndarray, pd.Series]:
    """Converts our suspected dates in dmy format to %Y-%m-%d

    Parameters
    ----------
    dates: str or array-like format
        A character or an array-like of suspected dates
    tz: str
        Time zone indicator. If None (default), a Date object is returned. Otherwise a POSIXct with time zone attribute set to tz.

    Returns
    -------
    Our suspected dates converted to %Y-%d-%m format
    """
    if tz is not None:
        zone = timezone(tz)
    if isinstance(dates, pd.Series):
        return_date = pd.to_datetime(dates, dayfirst=True, yearfirst=False)
        if tz is not None:
            return_date = return_date.dt.tz_localize(tz=zone)
    elif isinstance(dates, (np.ndarray, np.generic)):
        if tz is None:
            return_date = np.array(
                [dateutil.parser.parse(date, dayfirst=True, yearfirst=False).strftime("%Y-%m-%d") for date in dates],
                dtype="datetime64",
            )
        else:
            return_date = (
                pd.to_datetime(pd.Series(dates), dayfirst=True, yearfirst=False)
                .dt.tz_localize(tz=zone)
                .to_numpy(dtype="datetime64")
            )
    elif isinstance(dates, (list, tuple)):
        if tz is None:
            return_date = [
                dateutil.parser.parse(date, dayfirst=True, yearfirst=False).strftime("%Y-%m-%d") for date in dates
            ]
        else:
            return_date = [
                zone.localize(dateutil.parser.parse(date, dayfirst=True, yearfirst=False)).strftime("%Y-%m-%d %Z")
                for date in dates
            ]
    elif isinstance(dates, str):
        if tz is None:
            return_date = dateutil.parser.parse(dates, dayfirst=True, yearfirst=False).strftime("%Y-%m-%d")
        else:
            return_date = zone.localize(dateutil.parser.parse(dates, dayfirst=True, yearfirst=False)).strftime(
                "%Y-%m-%d %Z"
            )
    else:
        raise TypeError("Cannot identify date variable")
    return return_date


def _dym(dates: str):
    current_year = date.today().year
    month_cal = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    if dates.upper().isupper():
        all_numerics = re.findall(r"\d+", dates)
        all_letters = re.findall(r"[a-zA-Z]+", dates)
        month = month_cal[all_letters[0][0:3].capitalize()]
        if len(all_numerics) == 2:
            day = all_numerics[0]
            year = all_numerics[1]
        elif len(all_numerics) == 1:
            if len(all_numerics[0]) in [3, 5]:
                day = all_numerics[0][0:1]
                year = all_numerics[0][1:]
            elif len(all_numerics[0]) in [4, 6]:
                day = all_numerics[0][0:2]
                year = all_numerics[0][2:]
            else:
                raise ValueError("All formats failed to parse. No formats found.")
        else:
            raise ValueError("All formats failed to parse. No formats found.")
    else:
        if "/" in dates:
            date_split = dates.split("/")
        elif "-" in dates:
            date_split = dates.split("-")
        elif "." in dates:
            date_split = dates.split(".")
        elif " " in dates:
            date_split = dates.split(" ")
        else:
            if len(dates) == 4:
                date_split = [dates[1], dates[1:3], dates[3]]
            elif len(dates) == 6:
                date_split = [dates[0:2], dates[2:4], dates[4:]]
            elif len(dates) == 8:
                date_split = [dates[0:2], dates[2:6], dates[6:]]
            else:
                raise ValueError("All formats failed to parse. No formats found.")
        day, year, month = date_split[0], date_split[1], date_split[2]
    if len(year) == 2:
        if int(year) > current_year:
            year = "19" + year
        else:
            year = "20" + year
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    return datetime.strptime("{}-{}-{}".format(year, month, day), "%Y-%m-%d")


def dym(
    dates: Union[str, Sequence[str], np.ndarray, pd.Series], tz: str = None
) -> Union[str, Sequence[str], np.ndarray, pd.Series]:
    """Converts our suspected dates in dym format to %Y-%m-%d

    Parameters
    ----------
    dates: str or array-like format
        A character or an array-like of suspected dates
    tz: str
        Time zone indicator. If None (default), a Date object is returned. Otherwise a POSIXct with time zone attribute set to tz.

    Returns
    -------
    Our suspected dates converted to %Y-%d-%m format
    """
    if tz is not None:
        zone = timezone(tz)
    if isinstance(dates, pd.Series):
        return_date = pd.to_datetime(dates.apply(lambda x: _dym(x).strftime("%Y-%m-%d")))
        if tz is not None:
            return_date = return_date.dt.tz_localize(tz=zone)
    elif isinstance(dates, (np.ndarray, np.generic)):
        if tz is None:
            return_date = np.array(list(map(_dym, dates)), dtype="datetime64[D]")
        else:
            return_date = (
                pd.to_datetime(pd.Series(dates).apply(lambda x: _dym(x).strftime("%Y-%m-%d")))
                .dt.tz_localize(tz=zone)
                .to_numpy(dtype="datetime64")
            )
    elif isinstance(dates, (list, tuple)):
        if tz is None:
            return_date = [_dym(date).strftime("%Y-%m-%d") for date in dates]
        else:
            return_date = [zone.localize(_dym(date)).strftime("%Y-%m-%d %Z") for date in dates]
    elif isinstance(dates, str):
        if tz is None:
            return_date = _dym(dates).strftime("%Y-%m-%d")
        else:
            return_date = zone.localize(_dym(dates)).strftime("%Y-%m-%d %Z")
    else:
        raise TypeError("Cannot identify date variable")
    return return_date


# Year, month, day, hour, minute and second parsing
def ymd_hms(dates, tz=None):
    """Converts our suspected dates in ymd_hms format to %Y-%m-%d %h:%m:%s

    Parameters
    ----------
    dates: str or array-like format
        A character or an array-like of suspected dates
    tz: str
        Time zone indicator. If None (default), a Date object is returned. Otherwise a POSIXct with time zone attribute set to tz.

    Returns
    -------
    Our suspected dates converted to %Y-%m-%d %h:%m:%s format
    """
    if tz is not None:
        zone = timezone(tz)
    if isinstance(dates, pd.Series):
        ...
    elif isinstance(dates, str):
        if tz is None:
            return_date = dateutil.parser.parse(dates, dayfirst=True, yearfirst=False).strftime("%Y-%m-%d %H:%m:%S")
        else:
            return_date = zone.localize(dateutil.parser.parse(dates, dayfirst=True, yearfirst=False)).strftime(
                "%Y-%m-%d %Z"
            )
    else:
        raise TypeError("Cannot identify date variable")
    return return_date
