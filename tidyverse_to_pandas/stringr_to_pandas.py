"""Convert stringr string-manipulation syntax to pandas equivalents."""

import re
import unicodedata
from collections.abc import Sequence
from itertools import compress
from typing import cast

import numpy as np
import pandas as pd
from natsort import natsort_keygen

from tidyverse_to_pandas._optional_pyspark import ps


def _is_na(value: object) -> bool:
    """Return True if a single value is missing (None, NaN, pd.NA or NaT)."""
    return value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))


def _as_list(string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column, name: str) -> list:
    """Return the elements of our string container as a list, raising for unsupported containers."""
    if isinstance(string, str):
        return [string]
    elif isinstance(string, (list, tuple, np.ndarray, pd.Series)):
        return list(string)
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError(f"Cannot determine how to perform {name}")


def _wrap(string: str | Sequence[str] | np.ndarray | pd.Series, values: list) -> object:
    """Put element-wise results back into the same kind of container as our input, keeping any Series index."""
    if isinstance(string, str):
        return values[0]
    elif isinstance(string, (list, tuple)):
        return values
    elif isinstance(string, np.ndarray):
        return np.array(values)
    else:
        return pd.Series(values, index=string.index, name=string.name)


def _recycle(arg: object, length: int, name: str) -> list:
    """Recycle a scalar or length-1 argument to `length`, mirroring R's vector recycling rules."""
    if isinstance(arg, (str, bytes)) or arg is None or np.ndim(arg) == 0:
        return [arg] * length
    values = list(arg)
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"{name} must have length 1 or the same length as string")
    return values


# Character Manipulation


def str_length(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
) -> int | Sequence[int] | np.ndarray | pd.Series | ps.Column:
    """Calculate the length of each string.

    Parameters
    ----------
    string : str or list or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    The lengths of each string within our data
    """
    if isinstance(string, str):
        return len(string)
    elif isinstance(string, (list, tuple)):
        return [len(s) for s in string]
    elif isinstance(string, pd.Series):
        return string.str.len()
    elif isinstance(string, np.ndarray):
        return np.char.str_len(np.asarray(string))
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to calculate string length")


def str_sub(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    start: int,
    end: int | None = None,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Extract substrings from a character vector.

    One important note is that R and Python have vastly different indexing rules. R is [start, end] whereas Python is
    [start, end). Additionally, R has its indexes start at 1 and end at len(string), whereas Python has its indexes
    start at 0 and end at len(string) - 1. I figure that it would be too confusing to intermingle R indexing rules with
    Python, so this function follows Python indexing conventions.

    Parameters
    ----------
    string : str or list or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one
    start : int
        Our starting point
    end : int, default is None
        Our end point. If None, then it's the very end of the string

    Returns
    -------
    The substring of each string within our data
    """
    if isinstance(string, str):
        if end is None:
            return string[start:]
        else:
            return string[start:end]
    elif isinstance(string, (list, tuple)):
        if end is None:
            return [s[start:] for s in string]
        else:
            return [s[start:end] for s in string]
    elif isinstance(string, np.ndarray):
        # The way np.frompyfunc works is that it takes in a function (such as a lambda expression) and then you feed it
        # the parameters needed to run it. Essentially, it's a way of converting f(x) to something that numpy can apply
        # to all elements of the array
        if end is None:
            return np.frompyfunc(lambda x: x[start:], 1, 1)(string)
        else:
            return np.frompyfunc(lambda x: x[start:end], 1, 1)(string)
    elif isinstance(string, pd.Series):
        if end is None:
            return string.str.slice(start)
        else:
            return string.str.slice(start, end)
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to do string subset")


def str_dup(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    num_dupes: int | Sequence[int] | np.ndarray,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Duplicates our string.

    Parameters
    ----------
    string : str or list or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one
    num_dupes : int or list/tuple or numpy array
        Number of duplications. An integer implies equal number of duplications while a list/tuple or array specifies
        the number of duplications per entry

    Returns
    -------
    A duplicate of our string
    """
    if isinstance(string, str) and not isinstance(num_dupes, int):
        raise TypeError(f"Cannot determine number of duplications using type {type(num_dupes)}. Use integer instead")
    strings = _as_list(string, "string duplication")
    dupes = _recycle(num_dupes, len(strings), "num_dupes")
    return _wrap(string, [s if _is_na(s) else s * int(n) for s, n in zip(strings, dupes)])


def str_flatten(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    collapse: str = "",
    na_rm: bool = False,
) -> str | None:
    """Turn a collection of strings into a single, flattened string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    collapse : str
        String to insert between each piece
    na_rm : bool, default is False
        If True, missing values are dropped before flattening. Otherwise any missing value makes the result None.

    Returns
    -------
    A flattened string
    """
    if isinstance(string, str):
        return string
    strings = _as_list(string, "string flattening")
    if na_rm:
        strings = [s for s in strings if not _is_na(s)]
    elif any(_is_na(s) for s in strings):
        return None
    return collapse.join(strings)


def str_trunc(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    width: int,
    side: str = "right",
    ellipsis: str = "...",
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Truncate a character string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    width : int
        Maximum width of string, including the ellipsis. Strings no longer than this are returned unchanged.
    side : str, {right, left, center}, default is right
        Location of ellipsis that indicates content has been removed.
    ellipsis : str, default is ...
        Content of ellipsis that indicates content has been removed.

    Returns
    -------
    Our truncated string
    """
    side = side.casefold()
    if side not in ["right", "left", "center"]:
        raise ValueError("Cannot determine location of ellipsis.")
    keep = width - len(ellipsis)
    if keep < 0:
        raise ValueError("width must be at least as long as the ellipsis")

    def _trunc(s: str) -> str:
        if _is_na(s) or len(s) <= width:
            return s
        if side == "right":
            return s[:keep] + ellipsis
        elif side == "left":
            return ellipsis + (s[-keep:] if keep > 0 else "")
        else:
            # R keeps the extra character on the left when the remaining width is odd
            left, right = -(-keep // 2), keep // 2
            return s[:left] + ellipsis + (s[-right:] if right > 0 else "")

    return _wrap(string, [_trunc(s) for s in _as_list(string, "string truncation")])


def str_replace_na(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    replacement: str = "NA",
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Convert NaNs and None into strings.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    replacement : str
        A single string.

    Returns
    -------
    Our string, but with NaNs and None replaced as strings
    """
    if _is_na(string):
        return replacement
    elif isinstance(string, str):
        return string
    elif isinstance(string, (list, tuple)):
        return [replacement if _is_na(s) else s for s in string]
    elif isinstance(string, np.ndarray):
        return np.array([replacement if _is_na(s) else s for s in string], dtype=object)
    elif isinstance(string, pd.Series):
        return string.fillna(replacement)
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to convert NAs/None to string")


def str_unique(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    strength: int = 1,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Keep unique strings only, in the order they first appear.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        A character vector to return unique strings. A single str is de-duplicated character by character.
    strength : [1, 2, 3, 4]
        Collation strength, following ICU: `1` ignores both case and accents, `2` ignores case but not accents, and
        `3` and `4` compare strings exactly (after canonical Unicode normalisation)

    Returns
    -------
    All unique strings, keeping the first occurrence of each
    """
    if strength not in [1, 2, 3, 4]:
        raise ValueError("Cannot determine strength of collation")

    def _key(s: str) -> object:
        if _is_na(s):
            return None
        s = unicodedata.normalize("NFC", s)
        if strength == 1:
            return _remove_accents(s).casefold()
        elif strength == 2:
            return s.casefold()
        return s

    elements = list(string) if isinstance(string, str) else _as_list(string, "str_unique")
    seen: set = set()
    keep = []
    for position, s in enumerate(elements):
        key = _key(s)
        if key not in seen:
            seen.add(key)
            keep.append(position)
    if isinstance(string, str):
        return "".join(elements[i] for i in keep)
    elif isinstance(string, (list, tuple)):
        return [elements[i] for i in keep]
    elif isinstance(string, np.ndarray):
        return string[np.array(keep, dtype=int)]
    else:
        return string.iloc[keep]


# Case Transformations


def str_to_upper(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Convert all string to UPPERCASE.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    All of our strings in uppercase
    """
    if isinstance(string, str):
        return string.upper()
    elif isinstance(string, (list, tuple)):
        return [s.upper() for s in string]
    elif isinstance(string, np.ndarray):
        return np.char.upper(string)
    elif isinstance(string, pd.Series):
        return string.str.upper()
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to do string uppercase")


def str_to_title(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Convert all strings to title form.

    Each word is capitalised and the rest of the word lowercased. Apostrophes inside a word (e.g. "they're") do not
    start a new word, while other punctuation (e.g. "hello-world") does.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    All of our strings in title form
    """
    word = re.compile(r"\w+(?:['’]\w+)*")

    def _title(s: str) -> str:
        if _is_na(s):
            return s
        return word.sub(lambda m: m.group(0).capitalize(), s)

    return _wrap(string, [_title(s) for s in _as_list(string, "string titleization")])


def str_to_lower(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    locale: str = "en",
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Convert all of our strings into lowercase.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    locale : str, default is en
        Used to distinguish if we're dealing with non-English strings, such as Greek or Chinese

    Returns
    -------
    All of our strings in lowercase
    """
    if isinstance(string, str):
        if locale == "en":
            return string.casefold()
        else:
            return string.lower()
    elif isinstance(string, (list, tuple)):
        if locale == "en":
            return [s.casefold() for s in string]
        else:
            return [s.lower() for s in string]
    elif isinstance(string, np.ndarray):
        if locale == "en":
            return np.array(list(map(lambda v: v.casefold(), string)))
        else:
            return np.char.lower(string)
    elif isinstance(string, pd.Series):
        if locale == "en":
            return string.str.casefold()
        else:
            return string.str.lower()
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to lower strings")


def str_to_sentence(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Convert all of our strings into sentence format.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    All of our strings in sentence format
    """
    if isinstance(string, str):
        return string.capitalize()
    elif isinstance(string, (tuple, list)):
        return [s.capitalize() for s in string]
    elif isinstance(string, np.ndarray):
        return np.char.capitalize(string)
    elif isinstance(string, pd.Series):
        return string.str.capitalize()
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to capitalize string")


# String Ordering and Equality
def _order_positions(values: list, decreasing: bool, na_last: bool | None, numeric: bool) -> list[int]:
    """Return the positions that sort `values`, placing missing values according to `na_last`."""
    missing = [i for i, v in enumerate(values) if _is_na(v)]
    present = [i for i, v in enumerate(values) if not _is_na(v)]
    key = natsort_keygen() if numeric else (lambda v: v)
    present.sort(key=lambda i: key(values[i]), reverse=decreasing)
    if na_last is None:
        return present
    return present + missing if na_last else missing + present


def str_order(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    decreasing: bool = False,
    na_last: bool | None = True,
    numeric: bool = False,
) -> int | Sequence[int] | np.ndarray | pd.Series | ps.Column:
    """Return the positions (0-based) that would sort our strings.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    decreasing : bool, default is False
        If False, sorts from lowest to highest; if True sorts from highest to lowest.
    na_last : bool or None, default is True
        Where should NA go? True at the end, False at the beginning, None dropped. This applies regardless of
        `decreasing`.
    numeric : bool, default is False
        If True, will sort digits numerically, instead of as strings.

    Returns
    -------
    The positions of our strings in sorted order
    """
    if isinstance(string, str):
        return 0
    positions = _order_positions(_as_list(string, "string ordering"), decreasing, na_last, numeric)
    if isinstance(string, (list, tuple)):
        return positions
    elif isinstance(string, np.ndarray):
        return np.array(positions, dtype=int)
    else:
        return pd.Series(positions, dtype="int64")


def str_sort(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    decreasing: bool = False,
    na_last: bool | None = True,
    numeric: bool = False,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Sorts our strings based off the results of str_order.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    decreasing : bool, default is False
        If False, sorts from lowest to highest; if True sorts from highest to lowest.
    na_last : bool or None, default is True
        Where should NA go? True at the end, False at the beginning, None dropped.
    numeric : bool, default is False
        If True, will sort digits numerically, instead of as strings.

    Returns
    -------
    Our sorted string. A Series keeps the original index labels of each value.
    """
    if isinstance(string, str):
        return string
    positions = _order_positions(_as_list(string, "string sorting"), decreasing, na_last, numeric)
    if isinstance(string, (list, tuple)):
        return [string[i] for i in positions]
    elif isinstance(string, np.ndarray):
        return string[np.array(positions, dtype=int)]
    else:
        return string.iloc[positions]


def str_equal(
    x: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    y: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    ignore_case: bool = False,
) -> bool | Sequence[bool] | np.ndarray | pd.Series | ps.Column:
    """Determine if two strings are equivalent.

    Parameters
    ----------
    x, y : str or list/tuple or numpy array or pandas Series or pyspark column
        A pair of character vectors
    ignore_case : bool, default=False
        Ignore case when comparing strings?

    Returns
    -------
    Whether or not the strings are equal
    """
    if type(x) is not type(y):
        raise TypeError("x and y must be of the same type")
    if isinstance(x, str):
        y_str = cast(str, y)
        if ignore_case:
            return unicodedata.normalize("NFC", x.casefold()) == unicodedata.normalize("NFC", y_str.casefold())
        else:
            return unicodedata.normalize("NFC", x) == unicodedata.normalize("NFC", y_str)
    elif isinstance(x, pd.Series):
        x_series = cast(pd.Series, x)
        y_series = cast(pd.Series, y)
        if ignore_case:
            return pd.Series(
                np.where(
                    x_series.str.casefold().apply(lambda xi: unicodedata.normalize("NFC", xi))
                    == y_series.str.casefold().apply(lambda yi: unicodedata.normalize("NFC", yi)),
                    True,
                    False,
                )
            )
        else:
            return pd.Series(
                np.where(
                    x_series.apply(lambda xi: unicodedata.normalize("NFC", xi))
                    == y_series.apply(lambda yi: unicodedata.normalize("NFC", yi)),
                    True,
                    False,
                )
            )
    elif isinstance(x, (list, tuple)):
        if ignore_case:
            return [
                unicodedata.normalize("NFC", i.casefold()) == unicodedata.normalize("NFC", j.casefold())
                for i, j in zip(x, y)
            ]
        else:
            return [unicodedata.normalize("NFC", i) == unicodedata.normalize("NFC", j) for i, j in zip(x, y)]
    elif isinstance(x, np.ndarray):
        x_arr = cast(np.ndarray, x)
        y_arr = cast(np.ndarray, y)
        if ignore_case:
            return np.vectorize(
                lambda xi, yi: unicodedata.normalize("NFC", xi.upper()) == unicodedata.normalize("NFC", yi.upper())
            )(x_arr, y_arr)
        else:
            return np.vectorize(lambda xi, yi: unicodedata.normalize("NFC", xi) == unicodedata.normalize("NFC", yi))(
                x_arr, y_arr
            )
    else:
        raise TypeError("Cannot determine how to compare strings")


# Whitespace Manipulation


def str_pad(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    width: int | Sequence[int] | np.ndarray | pd.Series | ps.Column,
    side: str = "right",
    pad: str | Sequence[str] | np.ndarray | pd.Series | ps.Column = " ",
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Pad a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    width : int or list/tuple/numpy array/pandas Series
        Minimum width of padded strings
    side : str, {left, right, both}, default is right
        Side on which padding character is added. "center" is accepted as an alias for "both"; when the padding is
        uneven, the extra character goes on the right, as in R.
    pad : str or list/tuple/numpy array/pandas Series, default is " "
        Single padding character (default is a space).

    Returns
    -------
    Our padded string
    """
    side = side.casefold()
    if side == "center":
        side = "both"
    if side not in ["right", "left", "both"]:
        raise ValueError("Cannot determine where to pad string")

    def _pad(s: str, w: int, p: str) -> str:
        if _is_na(s):
            return s
        if side == "left":
            return s.rjust(w, p)
        elif side == "right":
            return s.ljust(w, p)
        total = max(w - len(s), 0)
        return p * (total // 2) + s + p * (total - total // 2)

    if isinstance(string, str):
        if isinstance(width, int) and isinstance(pad, str):
            return _pad(string, width, pad)
        # A single string with vectorised width/pad becomes a list, one entry per width/pad
        length = len(pad) if isinstance(width, int) else len(width)
        strings = [string] * length
        container: str | Sequence[str] | np.ndarray | pd.Series = strings
    else:
        strings = _as_list(string, "string padding")
        container = string
    widths = _recycle(width, len(strings), "width")
    pads = _recycle(pad, len(strings), "pad")
    return _wrap(container, [_pad(s, int(w), p) for s, w, p in zip(strings, widths, pads)])


def str_trim(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    side: str = "both",
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Remove leading and trailing whitespace in our string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    side : str, {left, right, both}, default is both
        Side on which to remove whitespace: "left" removes leading whitespace, "right" removes trailing whitespace.

    Returns
    -------
    Our trimmed string
    """
    side = side.casefold()
    if side not in ["left", "right", "both"]:
        raise ValueError("Cannot determine method of trimming")
    if isinstance(string, str):
        if side == "both":
            return string.strip()
        elif side == "left":
            return string.lstrip()
        else:
            return string.rstrip()
    elif isinstance(string, (list, tuple)):
        if side == "both":
            return [s.strip() for s in string]
        elif side == "left":
            return [s.lstrip() for s in string]
        else:
            return [s.rstrip() for s in string]
    elif isinstance(string, np.ndarray):
        if side == "both":
            return np.char.strip(string)
        elif side == "left":
            return np.char.lstrip(string)
        else:
            return np.char.rstrip(string)
    elif isinstance(string, pd.Series):
        if side == "both":
            return string.str.strip()
        elif side == "left":
            return string.str.lstrip()
        else:
            return string.str.rstrip()
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise ValueError("Cannot determine method for trimming strings")


def str_squish(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Squishes a string by removing leading and trailing whitespace as well as repeat whitespace inside a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    Our squished string
    """
    if not isinstance(string, (str, list, tuple, np.ndarray, pd.Series, ps.Column)):
        raise ValueError("Cannot determine method of squishing strings")
    squished = [s if _is_na(s) else " ".join(s.split()) for s in _as_list(string, "string squishing")]
    return _wrap(string, squished)


# String Pattern Matching


def str_detect(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str,
    negate: bool = False,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Determine if each string contains the regular expression.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for, interpreted as a regular expression.
    negate : bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Whether our pattern could be found within the string. Missing strings give None (NaN in a Series).
    """
    match: bool | list | np.ndarray | pd.Series
    if isinstance(string, str):
        match = (re.search(pattern, string) is None) if negate else (re.search(pattern, string) is not None)
    elif isinstance(string, (list, tuple)):
        match = [None if _is_na(s) else (re.search(pattern, s) is not None) != negate for s in string]
    elif isinstance(string, np.ndarray):
        # Unfortunately, numpy has no native methods for doing re.search, so you'll need to rely on either list
        # comprehension or pandas Series in order to get it to run properly
        if negate:
            match = (~pd.Series(string).str.contains(pattern, regex=True)).to_numpy()
        else:
            match = pd.Series(string).str.contains(pattern, regex=True).to_numpy()
    elif isinstance(string, pd.Series):
        match = string.str.contains(pattern, regex=True)
        if negate:
            match = ~match
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to do string detection")
    return match


def str_count(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str | Sequence,
) -> int | Sequence[int] | np.ndarray | pd.Series | ps.Column:
    """Count instances of pattern occurring within string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str or list
        Pattern to look for, interpreted as a regular expression. Either a single pattern or one per string.

    Returns
    -------
    The number of non-overlapping instances our pattern was found within the string
    """
    if isinstance(string, str) and not isinstance(pattern, str):
        raise TypeError("pattern must be str when string is str")
    strings = _as_list(string, "string count")
    patterns = _recycle(pattern, len(strings), "pattern")
    counts = [None if _is_na(s) else len(re.findall(p, s)) for s, p in zip(strings, patterns)]
    return _wrap(string, counts)


def str_subset(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str,
    negate: bool = False,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Filter our string based on str_detect.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    negate : bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Our filtered string
    """
    if isinstance(string, str):
        if str_detect(string, pattern, negate):
            return string
        else:
            return ""
    elif isinstance(string, (list, tuple)):
        return list(compress(string, str_detect(string, pattern, negate)))
    elif isinstance(string, np.ndarray):
        mask_arr = cast(np.ndarray, str_detect(string, pattern, negate))
        return string[mask_arr]
    elif isinstance(string, pd.Series):
        mask_series = cast(pd.Series, str_detect(string, pattern, negate))
        return string[mask_series]
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise TypeError("Cannot determine how to do string subset")


def str_which(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str,
    negate: bool = False,
) -> int | Sequence[int] | np.ndarray | pd.Series | ps.Column | None:
    """Filter our string based on str_detect.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    negate : bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Indices of our filtered string
    """
    string_loc = str_detect(string, pattern, negate)
    if isinstance(string, str):
        if string_loc:
            return 0
        else:
            return None
    elif isinstance(string, (list, tuple)):
        return [i for i, val in enumerate(string_loc) if val]
    elif isinstance(string, np.ndarray):
        return np.where(string_loc)[0]
    elif isinstance(string, pd.Series):
        string_loc_series = cast(pd.Series, string_loc)
        return pd.Series(string[string_loc_series.isin([True])].index)
    elif isinstance(string, ps.Column):
        raise NotImplementedError("PySpark support not yet implemented")
    else:
        raise ValueError("Cannot determine how to do string which")


def _str_replace(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str | Sequence[str],
    replacement: str | Sequence[str] | None,
    how: str = "all",
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    r"""Replace either the first or all instance(s) of pattern in string with replacement.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str or list
        Pattern to look for, interpreted as a regular expression. Either a single pattern or one per string.
    replacement : str or list or None
        A character vector of replacements. Should be either length one, or the same length as string.
        References of the form \1, \2, etc will be replaced with the contents of the respective matched group.
        A missing replacement (None/NaN) turns every string that matches the pattern into that missing value.
    how : str, default is 'all'
        Whether we are replacing all matches or only the first

    Returns
    -------
    The string, but with either the first or all matched patterns replaced
    """
    count = 0 if how.casefold() == "all" else 1
    if isinstance(string, str) and not isinstance(pattern, str):
        raise TypeError("pattern must be str when string is str")
    strings = _as_list(string, "string replace")
    patterns = _recycle(pattern, len(strings), "pattern")
    replacements = _recycle(replacement, len(strings), "replacement")

    def _replace(s: str, p: str, r: str | None) -> str | None:
        if _is_na(s):
            return s
        if _is_na(r):
            return r if re.search(p, s) is not None else s
        return re.sub(p, r, s, count=count)

    return _wrap(string, [_replace(s, p, r) for s, p, r in zip(strings, patterns, replacements)])


def str_replace(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str | Sequence[str],
    replacement: str | Sequence[str],
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Replace the first instance of pattern in string with replacement."""
    return _str_replace(string, pattern, replacement, how="first")


def str_replace_all(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str | Sequence[str],
    replacement: str | Sequence[str],
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Replace all instances of pattern in string with replacement."""
    return _str_replace(string, pattern, replacement, how="all")


def str_remove(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column, pattern: str
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Remove the first pattern in our string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").

    Returns
    -------
    Our string, but with the first pattern removed
    """
    return str_replace(string, pattern, "")


def str_remove_all(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column, pattern: str
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Remove all patterns in our string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").

    Returns
    -------
    Our string, but with all patterns removed
    """
    return str_replace_all(string, pattern, "")


def _split_pieces(s: str, pattern: str, n: int) -> list:
    """Split a single string on a regex into at most `n` pieces (all pieces when n < 1), mirroring stringr."""
    if _is_na(s):
        return [None]
    if n == 1:
        return [s]
    if pattern == "":
        # An empty pattern splits into characters, with the last piece holding any remainder
        pieces = list(s)
        if n > 1 and len(pieces) > n:
            pieces = pieces[: n - 1] + ["".join(pieces[n - 1 :])]
        return pieces
    return re.split(pattern, s, maxsplit=n - 1 if n > 1 else 0)


def _pad_pieces(pieces: list[list], length: int) -> list[list]:
    """Pad each list of pieces to `length` with "" (or None for a missing string), as str_split_fixed does."""
    return [p + ([None] if p == [None] else [""]) * (length - len(p)) for p in pieces]


def str_split(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str = " ",
    n: int = -1,
    simplify: bool = False,
) -> Sequence[str] | np.ndarray | pd.Series | ps.Column:
    """Split a string into pieces.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str, default is " "
        Pattern to split on, interpreted as a regular expression for every input type. An empty pattern splits into
        individual characters.
    n : int, default is -1
        Maximum number of pieces to return; the last piece holds the rest of the string. -1 returns all pieces.
    simplify : bool, default is False
        If False, the default, returns a list of character vectors. If True, returns a character matrix, padding
        short rows with "".

    Returns
    -------
    Our string, but split along the pattern
    """
    if isinstance(string, str):
        return _split_pieces(string, pattern, n)
    pieces = [_split_pieces(s, pattern, n) for s in _as_list(string, "string splitting")]
    if simplify:
        pieces = _pad_pieces(pieces, max(map(len, pieces), default=0))
    if isinstance(string, (list, tuple)):
        return pieces
    elif isinstance(string, np.ndarray):
        if simplify:
            return np.array(pieces)
        split_arr = np.empty(len(pieces), dtype=object)
        split_arr[:] = pieces
        return split_arr
    elif simplify:
        return pd.DataFrame(pieces, index=string.index)
    else:
        return pd.Series(pieces, index=string.index, name=string.name)


def str_split_fixed(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str = " ",
    n: int = -1,
) -> Sequence[str] | np.ndarray | pd.DataFrame | ps.DataFrame:
    """Split strings into a character matrix with exactly `n` columns.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str, default is " "
        Pattern to split on, interpreted as a regular expression.
    n : int, default is -1
        Number of pieces (columns) to return. Strings with fewer pieces are padded with "", and the last piece holds
        the remainder of longer strings. -1 uses as many columns as the longest split.

    Returns
    -------
    A matrix of character splits
    """
    if isinstance(string, str):
        pieces = _split_pieces(string, pattern, n)
        return _pad_pieces([pieces], max(n, len(pieces)))[0]
    split = str_split(string, pattern, n, simplify=True)
    if n < 1:
        return split
    # Pad out to exactly n columns, even if no string had that many pieces
    if isinstance(split, pd.DataFrame):
        for col in range(len(split.columns), n):
            split[col] = ["" if not _is_na(s) else None for s in string]
        return split
    rows = _pad_pieces([list(row) for row in split], n)
    return np.array(rows) if isinstance(string, np.ndarray) else rows


def str_split_n(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str = " ",
    n: int = 0,
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column | None:
    """Return the nth (0-based) piece of each split string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str, default is " "
        Pattern to split on, interpreted as a regular expression.
    n : int, default is 0
        The 0-based index of the piece to return. Strings with too few pieces give a missing value (None for str and
        list inputs, NaN for numpy arrays and Series).

    Returns
    -------
    A character vector containing the nth piece of each split string.
    """

    def _nth(s: str) -> str | None:
        pieces = _split_pieces(s, pattern, -1)
        return pieces[n] if n < len(pieces) else None

    if isinstance(string, str):
        return _nth(string)
    values = [_nth(s) for s in _as_list(string, "string splitting for fixed n")]
    if isinstance(string, (list, tuple)):
        return values
    elif isinstance(string, np.ndarray):
        return np.array([np.nan if v is None else v for v in values], dtype=object)
    else:
        return pd.Series(values, index=string.index, name=string.name)


def str_starts(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str,
    negate: bool = False,
) -> bool | Sequence[bool] | np.ndarray | pd.Series | ps.Column:
    """Detect the presence or absence of a pattern at the beginning of a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for, interpreted as a regular expression.
    negate : bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Either a bool or a container of bools indicating whether we can find our pattern at the start of the string
    """
    # Group the pattern so alternation ("a|b") is anchored as a whole
    return str_detect(string, "^(?:" + pattern + ")", negate)


def str_ends(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str,
    negate: bool = False,
) -> bool | Sequence[bool] | np.ndarray | pd.Series | ps.Column:
    """Detect the presence or absence of a pattern at the end of a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for, interpreted as a regular expression.
    negate : bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Either a bool or a container of bools indicating whether we can find our pattern at the end of the string
    """
    # Group the pattern so alternation ("a|b") is anchored as a whole
    return str_detect(string, "(?:" + pattern + ")$", negate)


def str_extract(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column, pattern: str
) -> str | Sequence[str] | np.ndarray | pd.Series | ps.Column | None:
    """Extract first matching patterns from a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for, interpreted as a regular expression.

    Returns
    -------
    First match within each string, or None if there is no match
    """
    if not isinstance(string, (str, list, tuple, np.ndarray, pd.Series, ps.Column)):
        raise TypeError("Cannot determine method of string extraction")
    regex = re.compile(pattern)

    def _extract(s: str) -> str | None:
        if _is_na(s):
            return None
        m = regex.search(s)
        return m.group(0) if m is not None else None

    return _wrap(string, [_extract(s) for s in _as_list(string, "string extraction")])


def str_extract_all(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column,
    pattern: str,
    simplify: bool = False,
) -> str | Sequence[str] | np.ndarray | pd.Series | pd.DataFrame | ps.Column:
    """Extract all matching patterns from a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for, interpreted as a regular expression. Whole matches are returned, even when the pattern
        contains capture groups (use str_match_all for the groups).
    simplify : bool, default is False
        If False, the default, returns one list of matches per string (a Series of lists for a Series). If True,
        returns a character matrix padded with "" (a DataFrame for a Series).

    Returns
    -------
    All non-overlapping matches within each string. A missing string gives [None].
    """
    regex = re.compile(pattern)

    def _extract_all(s: str) -> list:
        if _is_na(s):
            return [None]
        return [m.group(0) for m in regex.finditer(s)]

    if isinstance(string, str):
        return _extract_all(string)
    matches = [_extract_all(s) for s in _as_list(string, "string extraction")]
    if simplify:
        matches = _pad_pieces(matches, max(map(len, matches), default=0))
    if isinstance(string, (list, tuple)):
        return matches
    elif isinstance(string, np.ndarray):
        if simplify:
            return np.array(matches, dtype=object)
        match_arr = np.empty(len(matches), dtype=object)
        match_arr[:] = matches
        return match_arr
    elif simplify:
        return pd.DataFrame(matches, index=string.index)
    else:
        return pd.Series(matches, index=string.index, name=string.name)


def str_match(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column, pattern: str
) -> Sequence[str] | np.ndarray | pd.DataFrame | ps.DataFrame | None:
    """Extract the first match and its capture groups from a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for, interpreted as a regular expression with capture groups.

    Returns
    -------
    For each string, the whole match followed by each capture group, or all None if there is no match. A str gives
    a list (None if no match), a list gives a list of lists, a numpy array gives a 2-D object array, and a Series
    gives a DataFrame with columns "whole_match", 1, 2, ... sharing the Series' index.
    """
    regex = re.compile(pattern)
    width = regex.groups + 1

    def _match(s: str) -> list:
        m = None if _is_na(s) else regex.search(s)
        return [None] * width if m is None else [m.group(0), *m.groups()]

    if isinstance(string, str):
        row = _match(string)
        return None if row[0] is None else row
    rows = [_match(s) for s in _as_list(string, "string matching")]
    if isinstance(string, (list, tuple)):
        return rows
    elif isinstance(string, np.ndarray):
        return np.array(rows, dtype=object).reshape(len(rows), width)
    else:
        return pd.DataFrame(rows, index=string.index, columns=["whole_match", *range(1, width)])


def str_match_all(
    string: str | Sequence[str] | np.ndarray | pd.Series | ps.Column, pattern: str
) -> Sequence[str] | np.ndarray | pd.Series | None:
    """Extract every match and its capture groups from a string.

    Parameters
    ----------
    string : str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern : str
        Pattern to look for, interpreted as a regular expression with capture groups.

    Returns
    -------
    One matrix per string, with a row per match holding the whole match followed by each capture group. A string
    with no match gives an empty matrix, and a missing string gives a single row of None. A str gives a list of rows,
    a list gives a list of these, a numpy array gives an object array of 2-D arrays, and a Series gives a Series of
    lists of rows sharing its index.
    """
    regex = re.compile(pattern)
    width = regex.groups + 1

    def _match_all(s: str) -> list:
        if _is_na(s):
            return [[None] * width]
        return [[m.group(0), *m.groups()] for m in regex.finditer(s)]

    if isinstance(string, str):
        return _match_all(string)
    matches = [_match_all(s) for s in _as_list(string, "string matching")]
    if isinstance(string, (list, tuple)):
        return matches
    elif isinstance(string, np.ndarray):
        match_arr = np.empty(len(matches), dtype=object)
        match_arr[:] = [np.array(m, dtype=object).reshape(len(m), width) for m in matches]
        return match_arr
    else:
        return pd.Series(matches, index=string.index, name=string.name)


# Combining Strings


def str_c(
    strings: Sequence | np.ndarray | pd.DataFrame | ps.DataFrame,
    sep: str = "",
    collapse: str | None = None,
) -> str | list | None:
    """Join multiple character vectors into a single character vector.

    Parameters
    ----------
    strings : list/tuple, numpy array or pandas DataFrame
        One or more character vectors. Each element of a list/tuple (or each row of a 2-D numpy array, or each
        element of a 1-D one) is one vector; a str element is a length-1 vector. Each column of a DataFrame is one
        vector. Vectors of length 1 are recycled to the longest length; other lengths must match.
    sep : str, default=""
        String to insert between input vectors
    collapse : str, default=None
        Optional string used to combine outputs into a single string

    Returns
    -------
    If `collapse=None`, a list with length equal to the longest input. Else, a single string. Any missing value in a
    row makes that row None (and the collapsed result None).
    """
    if isinstance(strings, pd.DataFrame):
        vectors = [strings[col].tolist() for col in strings.columns]
    elif isinstance(strings, ps.DataFrame):
        raise NotImplementedError("PySpark support not yet implemented")
    elif isinstance(strings, (list, tuple, np.ndarray)):
        vectors = [
            [vector] if isinstance(vector, str) or _is_na(vector) else list(vector)
            for vector in (strings.tolist() if isinstance(strings, np.ndarray) else strings)
        ]
    else:
        raise TypeError("Cannot identify character vector")
    if not vectors or any(len(v) == 0 for v in vectors):
        return_string: list = []
    else:
        max_length = max(len(v) for v in vectors)
        columns = [_recycle(v, max_length, "each character vector") for v in vectors]
        return_string = [
            None if any(_is_na(value) for value in row) else sep.join(str(value) for value in row)
            for row in zip(*columns)
        ]
    if collapse is not None:
        return None if any(_is_na(s) for s in return_string) else collapse.join(return_string)
    return return_string


def _remove_accents(input_str: str) -> str:
    """Remove all accents from a string.

    Parameters
    ----------
    input_str : string
        The string we wish to remove accents from

    Returns
    -------
    Our string, but with all accents removed
    """
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
