"""Convert lubridate date-parsing syntax to pandas equivalents."""

import re
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np
import pandas as pd
from pytz import timezone

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_FULL_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_TOKEN = re.compile(r"\d+|[A-Za-z]+")
_ORDINAL_SUFFIX = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)
_TIME = re.compile(r"[ T]+(\d{1,2}):(\d{1,2}):(\d{1,2})(\.\d+)?\s*$")

DateInput = str | Sequence[str] | np.ndarray | pd.Series


def _month_from_name(token: str) -> int | None:
    """Return the month number for a month name or abbreviation (e.g. "Jan", "Sept", "January"), else None."""
    lowered = token.lower()
    if len(lowered) < 3:
        return None
    for number, full_name in enumerate(_FULL_MONTHS, start=1):
        if full_name.startswith(lowered) or (lowered == "sept" and number == 9):
            return number
    return None


def _expand_year(year: str) -> int:
    """Expand a two-digit year the way lubridate does: 00-68 become 20xx and 69-99 become 19xx."""
    value = int(year)
    if len(year) <= 2:
        value += 2000 if value <= 68 else 1900
    return value


def _split_compact(digits: str, fields: str) -> list[str] | None:
    """Split a run of digits with no separators (e.g. "20200102") into the given fields, in order."""
    if len(fields) == 3:
        widths = {8: (4, 2, 2), 6: (2, 2, 2), 4: (2, 1, 1)}.get(len(digits))
        if widths is None:
            return None
        year_size, other_size = widths[0], widths[1]
    else:
        # A month name was given separately, so the digits hold only the year and the day
        year_size = 4 if len(digits) >= 5 else 2
        other_size = len(digits) - year_size
        if other_size not in (1, 2):
            return None
    sizes = [year_size if field == "y" else other_size for field in fields]
    parts, start = [], 0
    for size in sizes:
        parts.append(digits[start : start + size])
        start += size
    return parts


def _parse_date(value: str, order: str) -> datetime | None:
    """Parse a single date whose year, month and day appear in ``order`` (e.g. "ymd"); None if it fails."""
    tokens = _TOKEN.findall(_ORDINAL_SUFFIX.sub("", value))
    # A month name can only be the month, so it fills that field wherever it appears
    month = None
    numerics = []
    for token in tokens:
        if token.isalpha():
            name_month = _month_from_name(token)
            if name_month is None or month is not None:
                return None
            month = name_month
        else:
            numerics.append(token)
    fields = order if month is None else order.replace("m", "")
    if len(numerics) == 1 and len(fields) > 1:
        numerics = _split_compact(numerics[0], fields)
        if numerics is None:
            return None
    if len(numerics) != len(fields):
        return None
    parts = dict(zip(fields, numerics))
    try:
        # Naive on purpose: the caller localizes to `tz` afterwards
        return datetime(_expand_year(parts["y"]), month or int(parts["m"]), int(parts["d"]))  # noqa: DTZ001
    except ValueError:
        return None


def _parse_date_time(value: str, order: str) -> datetime | None:
    """Parse a single date-time whose date is in ``order`` followed by hours, minutes and seconds."""
    match = _TIME.search(value)
    if match is None:
        return None
    parsed_date = _parse_date(value[: match.start()], order)
    if parsed_date is None:
        return None
    hour, minute, second, fraction = match.groups()
    try:
        return parsed_date.replace(
            hour=int(hour),
            minute=int(minute),
            second=int(second),
            microsecond=round(float(fraction) * 1_000_000) if fraction else 0,
        )
    except ValueError:
        return None


def _convert(
    dates: DateInput, tz: str | None, parser: Callable[[str], datetime | None], fmt: str, unit: str
) -> str | list | np.ndarray | pd.Series:
    """Apply ``parser`` to every element of ``dates`` and return it in the shape of the input.

    Values that fail to parse become missing (None for strings and lists, NaT otherwise) with a warning, as lubridate
    returns NA.
    """
    zone = timezone(tz) if tz is not None else None
    failures = 0

    def parse(value):
        nonlocal failures
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None
        parsed = parser(str(value))
        if parsed is None:
            failures += 1
        return parsed

    def to_string(parsed: datetime | None) -> str | None:
        if parsed is None:
            return None
        if zone is None:
            return parsed.strftime(fmt)
        return zone.localize(parsed).strftime(f"{fmt} %Z")

    if isinstance(dates, str):  # also covers numpy string scalars (np.str_)
        result = to_string(parse(dates))
    elif isinstance(dates, (list, tuple)):
        result = [to_string(parse(value)) for value in dates]
    elif isinstance(dates, (pd.Series, np.ndarray)):
        index = dates.index if isinstance(dates, pd.Series) else None
        parsed = pd.Series([parse(value) for value in dates], index=index, dtype="object")
        parsed = pd.to_datetime(parsed).astype("datetime64[ns]")
        if zone is not None:
            parsed = parsed.dt.tz_localize(tz=zone)
        if isinstance(dates, pd.Series):
            result = parsed
        elif zone is None:
            result = parsed.to_numpy(dtype=f"datetime64[{unit}]")
        else:
            # numpy datetimes cannot carry a time zone, so return the equivalent UTC instants
            result = parsed.dt.tz_convert(None).to_numpy()
    else:
        raise TypeError("Cannot identify date variable")
    if failures:
        warnings.warn(f"{failures} failed to parse.", stacklevel=3)
    return result


def ymd(dates: DateInput, tz: str | None = None) -> str | list | np.ndarray | pd.Series:
    """Parse dates whose year, month and day appear in ymd order.

    Parameters
    ----------
    dates : str, list, tuple, numpy array or pandas Series
        A string or an array-like of suspected dates. Any separator works, as do month names and compact digit
        strings (e.g. "20200102"). Two-digit years 00-68 become 20xx and 69-99 become 19xx, as in lubridate.
    tz : str, optional
        Time zone. If None (default), dates are returned without a time zone.

    Returns
    -------
    Strings and lists return "%Y-%m-%d" strings ("%Y-%m-%d %Z" when ``tz`` is given), numpy arrays return
    ``datetime64`` (as UTC instants when ``tz`` is given), and Series return ``datetime64[ns]`` (tz-aware when ``tz``
    is given). Dates that fail to parse are returned as missing, with a warning.
    """
    return _convert(dates, tz, lambda value: _parse_date(value, "ymd"), "%Y-%m-%d", "D")


def ydm(dates: DateInput, tz: str | None = None) -> str | list | np.ndarray | pd.Series:
    """Parse dates whose year, month and day appear in ydm order.

    Parameters
    ----------
    dates : str, list, tuple, numpy array or pandas Series
        A string or an array-like of suspected dates. Any separator works, as do month names and compact digit
        strings (e.g. "20200102"). Two-digit years 00-68 become 20xx and 69-99 become 19xx, as in lubridate.
    tz : str, optional
        Time zone. If None (default), dates are returned without a time zone.

    Returns
    -------
    Strings and lists return "%Y-%m-%d" strings ("%Y-%m-%d %Z" when ``tz`` is given), numpy arrays return
    ``datetime64`` (as UTC instants when ``tz`` is given), and Series return ``datetime64[ns]`` (tz-aware when ``tz``
    is given). Dates that fail to parse are returned as missing, with a warning.
    """
    return _convert(dates, tz, lambda value: _parse_date(value, "ydm"), "%Y-%m-%d", "D")


def mdy(dates: DateInput, tz: str | None = None) -> str | list | np.ndarray | pd.Series:
    """Parse dates whose year, month and day appear in mdy order.

    Parameters
    ----------
    dates : str, list, tuple, numpy array or pandas Series
        A string or an array-like of suspected dates. Any separator works, as do month names and compact digit
        strings (e.g. "20200102"). Two-digit years 00-68 become 20xx and 69-99 become 19xx, as in lubridate.
    tz : str, optional
        Time zone. If None (default), dates are returned without a time zone.

    Returns
    -------
    Strings and lists return "%Y-%m-%d" strings ("%Y-%m-%d %Z" when ``tz`` is given), numpy arrays return
    ``datetime64`` (as UTC instants when ``tz`` is given), and Series return ``datetime64[ns]`` (tz-aware when ``tz``
    is given). Dates that fail to parse are returned as missing, with a warning.
    """
    return _convert(dates, tz, lambda value: _parse_date(value, "mdy"), "%Y-%m-%d", "D")


def myd(dates: DateInput, tz: str | None = None) -> str | list | np.ndarray | pd.Series:
    """Parse dates whose year, month and day appear in myd order.

    Parameters
    ----------
    dates : str, list, tuple, numpy array or pandas Series
        A string or an array-like of suspected dates. Any separator works, as do month names and compact digit
        strings (e.g. "20200102"). Two-digit years 00-68 become 20xx and 69-99 become 19xx, as in lubridate.
    tz : str, optional
        Time zone. If None (default), dates are returned without a time zone.

    Returns
    -------
    Strings and lists return "%Y-%m-%d" strings ("%Y-%m-%d %Z" when ``tz`` is given), numpy arrays return
    ``datetime64`` (as UTC instants when ``tz`` is given), and Series return ``datetime64[ns]`` (tz-aware when ``tz``
    is given). Dates that fail to parse are returned as missing, with a warning.
    """
    return _convert(dates, tz, lambda value: _parse_date(value, "myd"), "%Y-%m-%d", "D")


def dmy(dates: DateInput, tz: str | None = None) -> str | list | np.ndarray | pd.Series:
    """Parse dates whose year, month and day appear in dmy order.

    Parameters
    ----------
    dates : str, list, tuple, numpy array or pandas Series
        A string or an array-like of suspected dates. Any separator works, as do month names and compact digit
        strings (e.g. "20200102"). Two-digit years 00-68 become 20xx and 69-99 become 19xx, as in lubridate.
    tz : str, optional
        Time zone. If None (default), dates are returned without a time zone.

    Returns
    -------
    Strings and lists return "%Y-%m-%d" strings ("%Y-%m-%d %Z" when ``tz`` is given), numpy arrays return
    ``datetime64`` (as UTC instants when ``tz`` is given), and Series return ``datetime64[ns]`` (tz-aware when ``tz``
    is given). Dates that fail to parse are returned as missing, with a warning.
    """
    return _convert(dates, tz, lambda value: _parse_date(value, "dmy"), "%Y-%m-%d", "D")


def dym(dates: DateInput, tz: str | None = None) -> str | list | np.ndarray | pd.Series:
    """Parse dates whose year, month and day appear in dym order.

    Parameters
    ----------
    dates : str, list, tuple, numpy array or pandas Series
        A string or an array-like of suspected dates. Any separator works, as do month names and compact digit
        strings (e.g. "20200102"). Two-digit years 00-68 become 20xx and 69-99 become 19xx, as in lubridate.
    tz : str, optional
        Time zone. If None (default), dates are returned without a time zone.

    Returns
    -------
    Strings and lists return "%Y-%m-%d" strings ("%Y-%m-%d %Z" when ``tz`` is given), numpy arrays return
    ``datetime64`` (as UTC instants when ``tz`` is given), and Series return ``datetime64[ns]`` (tz-aware when ``tz``
    is given). Dates that fail to parse are returned as missing, with a warning.
    """
    return _convert(dates, tz, lambda value: _parse_date(value, "dym"), "%Y-%m-%d", "D")


def ymd_hms(dates: DateInput, tz: str | None = None) -> str | list | np.ndarray | pd.Series:
    """Parse date-times in year, month, day, hour, minute, second order.

    Parameters
    ----------
    dates : str, list, tuple, numpy array or pandas Series
        A string or an array-like of suspected date-times, e.g. "2020-01-02 03:04:05" or "2020-01-02T03:04:05.5".
    tz : str, optional
        Time zone. If None (default), date-times are returned without a time zone.

    Returns
    -------
    Strings and lists return "%Y-%m-%d %H : %M:%S" strings ("%Y-%m-%d %H:%M:%S %Z" when ``tz`` is given), numpy arrays
    return ``datetime64`` (in UTC when ``tz`` is given), and Series return ``datetime64[ns]`` (tz-aware when ``tz`` is
    given). Values that fail to parse are returned as missing, with a warning.
    """
    return _convert(dates, tz, lambda value: _parse_date_time(value, "ymd"), "%Y-%m-%d %H:%M:%S", "us")
