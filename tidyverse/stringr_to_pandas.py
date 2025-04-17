import re
import unicodedata

from itertools import compress
from string import capwords
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyspark.sql as ps

from natsort import index_natsorted


# Character Manipulation


def str_length(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
) -> Union[int, Sequence[int], np.ndarray, pd.Series, ps.Column]:
    """Calculates the length of each string

    Parameters
    ----------
    string: str or list or numpy array or pandas Series or pyspark column
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
    elif isinstance(string, (np.ndarray, np.generic)):
        return np.char.str_len(string)
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to calculate string length")


def str_sub(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    start: int,
    end: Optional[int] = None,
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Extract substrings from a character vector

    One important note is that R and Python have vastly different indexing rules. R is [start, end] whereas Python is
    [start, end). Additionally, R has its indexes start at 1 and end at len(string), whereas Python has its indexes
    start at 0 and end at len(string) - 1. I figure that it would be too confusing to intermingle R indexing rules with
    Python, so this function follows Python indexing conventions.

    Parameters
    ----------
    string: str or list or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one
    start: int
        Our starting point
    end: int, default is None
        Our end point. If None, then it's the very end of the string

    Returns
    -------
    The substring of each string wtihin our data
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
    elif isinstance(string, (np.ndarray, np.generic)):
        # The way np.frompyfunc works is that it takes in a function (such as a lambda expression) and then you feed it
        # the parameters needed to run it. Essentially, it's a way of converting f(x) to something that numpy can apply
        # to all elements of the array
        if end is None:
            np.frompyfunc(lambda x: x[start:], 1, 1)(string)
        elif end > 0 and start > 0:
            b = string.view((str, 1)).reshape(len(string), -1)[:, start:end]
            return np.fromstring(b.tostring(), dtype=(str, end - start))
        else:
            return np.frompyfunc(lambda x: x[start:end], 1, 1)(string)
    elif isinstance(string, pd.Series):
        if end is None:
            return string.str.slice(start)
        else:
            return string.str.slice(start, end)
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to do string subset")


def str_dup(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    num_dupes: Union[int, Sequence[int], np.ndarray],
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Duplicates our string

    Parameters
    ----------
    string: str or list or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one
    num_dupes: int or list/tuple or numpy array
        Number of duplications. An integer implies equal number of duplications while a list/tuple or array specifies
        the number of duplications per entry

    Returns
    -------
    A duplicate of our string
    """
    if isinstance(string, str):
        if not isinstance(num_dupes, int):
            raise TypeError(
                "Cannot determine number of duplications using type {}. Use integer instead".format(type(num_dupes))
            )
        else:
            return string * num_dupes
    elif isinstance(string, (list, tuple)):
        if isinstance(num_dupes, int):
            return [s * num_dupes for s in string]
        else:
            duplicates = []
            for index in range(len(string)):
                duplicates.append(string[index] * num_dupes[index])
            return duplicates
    elif isinstance(string, (np.ndarray, np.generic)):
        # While I know I could follow a similar procedure above and simply return np.array(duplicates), it didn't feel
        # like in the spirit of numpy so I looked into using a numpy solution myself. Unfortunately, numpy has no way of
        # natively concatenating the values within an array unless they are all the same shape, which we cannot
        # guarantee, so it ends with us calling np.array on list comprehension
        if isinstance(num_dupes, int):
            cum_dupes = [num_dupes]
        else:
            cum_dupes = np.cumsum(num_dupes)[:-1]
        repeat_array = np.repeat(string, num_dupes)
        split_repeat = np.split(repeat_array, cum_dupes)
        if len(string) == 1:
            return np.array(["".join(row) for row in split_repeat])[:-1]
        else:
            return np.array(["".join(row) for row in split_repeat])
    elif isinstance(string, pd.Series):
        return string.str.repeat(repeats=num_dupes)
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to do string duplication")


def str_flatten(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    collapse: str = "",
) -> str:
    """Turns a collection of strings into a single, flattened string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    collapse: str
        String to insert between each piece

    Returns
    -------
    A flattened string
    """
    if isinstance(string, str):
        return string
    elif isinstance(string, (list, tuple)):
        return "{}".format(collapse).join([char for char in string])
    elif isinstance(string, (np.ndarray, np.generic)):
        # Ironically, we can call this for both pandas and lists, but I wanted to show the various ways of doing it
        # rather than simply calling the iterable.
        return "{}".format(collapse).join(string)
    elif isinstance(string, pd.Series):
        return "{}".format(collapse).join(string.values.flatten())


def str_trunc(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    width: int,
    side: str = "right",
    ellipsis: str = "...",
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Truncate a character string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    width: int
        Maximum width of string. In this case, the number of elements we want to display
    side: str, {right, left, center}, default is right
        Location of ellipsis that indicates content has been removed.
    ellipsis: str, default is ...
        Content of ellipsis that indicates content has been removed.

    Returns
    -------
    Our truncated string
    """
    if side.casefold() not in ["right", "left", "center"]:
        raise ValueError("Cannot determine location of ellipsis.")
    if isinstance(string, str):
        if side.casefold() == "right":
            return string[:width] + str(ellipsis)
        elif side.casefold() == "left":
            return str(ellipsis) + string[-width:]
        else:
            return string[: width // 2] + ellipsis + string[-width // 2 :]
    elif isinstance(string, (list, tuple)):
        if side.casefold() == "right":
            return [s[:width] + str(ellipsis) for s in string]
        elif side.casefold() == "left":
            return [str(ellipsis) + s[-width:] for s in string]
        else:
            return [s[: width // 2] + str(ellipsis) + s[-width // 2 :] for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        # The way np.frompyfunc works is that it takes in a function (such as a lambda expression) and then you feed it
        # the parameters needed to run it. Essentially, it's a way of converting f(x) to something that numpy can apply
        # to all elements of the array
        if side.casefold() == "right":
            return np.frompyfunc(lambda s: s[:width] + str(ellipsis), 1, 1)(string)
        elif side.casefold() == "left":
            return np.frompyfunc(lambda s: str(ellipsis) + s[-width:], 1, 1)(string)
        else:
            return np.frompyfunc(lambda s: s[: width // 2] + str(ellipsis) + s[-width // 2], 1, 1)(string)
    elif isinstance(string, pd.Series):
        if side.casefold() == "right":
            return string.str.slice(stop=width) + str(ellipsis)
        elif side.casefold() == "left":
            return string.str.slice(start=width) + str(ellipsis)
        else:
            return string.str.slice(stop=width // 2) + str(ellipsis) + string.str.slice(start=width // 2)
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to perform string truncation")


def str_replace_na(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    replacement: str = "NA",
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Converts NaNs and None into strings

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    replacement: str
        A single string.

    Returns
    -------
    Our string, but with NaNs and None replaced as strings
    """
    if string is None or string is np.nan:
        string = replacement
    elif isinstance(string, (list, tuple)):
        string = [replacement if s in [None, np.nan] else s for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        # numpy has no native way of detecting None except by using the == operator
        string[string == None] = replacement
        string[string != string] = replacement
    elif isinstance(string, pd.Series):
        string = string.fillna(replacement)
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to convert NAs/None to string")
    return string


def str_unique(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    strength: int = 1,
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Keeps unique strings only

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        A character vector to return unique strings
    strength: [1, 2, 3, 4]
        Single integer which defines collation strength; `1` for the most permissive collation rules, `4` for the
        strictest ones

    Returns
    -------
    All unique strings
    """
    if strength not in [1, 2, 3, 4]:
        raise ValueError("Cannot determine strength of collation")
    if isinstance(string, str):
        if strength in [3, 4]:
            return "".join(dict.fromkeys(string).keys())
        elif strength == 2:
            return "".join(dict.fromkeys(unicodedata.normalize("NFC", _remove_accents(string))).keys())
        else:
            str_array = np.array(list(string))
            return "".join(
                (
                    str_array[
                        np.sort(
                            np.unique(
                                np.fromiter(
                                    (unicodedata.normalize("NFC", _remove_accents(xi)).casefold() for xi in string),
                                    dtype=str_array.dtype,
                                ),
                                return_index=True,
                            )[1]
                        )
                    ]
                )
            )

    elif isinstance(string, (list, tuple)):
        if strength in [3, 4]:
            return list(dict.fromkeys(string).keys())
        elif strength == 2:
            return list(dict.fromkeys([unicodedata.normalize("NFC", _remove_accents(s)) for s in string]).keys())
        else:
            str_array = np.array(string)
            return list(
                str_array[
                    np.sort(
                        np.unique(
                            np.fromiter(
                                (unicodedata.normalize("NFC", _remove_accents(xi)).casefold() for xi in string),
                                dtype=str_array.dtype,
                            ),
                            return_index=True,
                        )[1]
                    )
                ]
            )
    elif isinstance(string, (np.ndarray, np.generic)):
        if strength in [3, 4]:
            return np.unique(string)
        elif strength == 2:
            return string[
                np.sort(
                    np.unique(
                        np.fromiter(
                            (unicodedata.normalize("NFC", _remove_accents(xi)) for xi in string),
                            dtype=string.dtype,
                        ),
                        return_index=True,
                    )[1]
                )
            ]
        else:
            return string[
                np.sort(
                    np.unique(
                        np.fromiter(
                            (unicodedata.normalize("NFC", _remove_accents(xi)).casefold() for xi in string),
                            dtype=string.dtype,
                        ),
                        return_index=True,
                    )[1]
                )
            ]
    elif isinstance(string, pd.Series):
        if strength in [3, 4]:
            return pd.Series(string.unique())
        elif strength == 2:
            indices = np.sort(
                np.unique(
                    np.fromiter(
                        (unicodedata.normalize("NFC", _remove_accents(xi)) for xi in string),
                        dtype="<U8",
                    ),
                    return_index=True,
                )[1]
            )
            series = string[indices]
            series.index = list(np.arange(len(series)))
        else:
            indices = np.sort(
                np.unique(
                    np.fromiter(
                        (unicodedata.normalize("NFC", _remove_accents(xi)).casefold() for xi in string),
                        dtype="<U8",
                    ),
                    return_index=True,
                )[1]
            )
            series = string[indices]
            series.index = np.arange(len(series))
        return series
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot identify value for str_unique")


# Case Transformations


def str_to_upper(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Converts all string to UPPERCASE

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    All of our strings in uppercase
    """
    if isinstance(string, str):
        return string.upper()
    elif isinstance(string, (list, tuple)):
        return [s.upper() for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        return np.char.upper(string)
    elif isinstance(string, pd.Series):
        return string.str.upper()
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to do string uppercase")


def str_to_title(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Converts all strings to title form

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    All of our strings in title form
    """
    if isinstance(string, str):
        return capwords(string)
    elif isinstance(string, (list, tuple)):
        return [s.title() for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        return np.char.title(string)
    elif isinstance(string, pd.Series):
        return string.str.title()
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine to how titalize strings")


def str_to_lower(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    locale: str = "en",
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Converts all of our strings into lowercase

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    locale: str, default is en
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
    elif isinstance(string, (np.ndarray, np.generic)):
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
        ...
    else:
        raise TypeError("Cannot determine how to lower strings")


def str_to_sentence(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Convert all of our strings into sentence format

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    All of our strings in sentence format
    """
    if isinstance(string, str):
        return string.capitalize()
    elif isinstance(string, (tuple, list)):
        return [s.capitalize() for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        return np.char.capitalize(string)
    elif isinstance(string, pd.Series):
        return string.str.capitalize()
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to capitalize string")


# String Ordering and Equality
def str_order(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    decreasing: bool = False,
    na_last: bool = True,
    numeric: bool = False,
) -> Union[int, Sequence[int], np.ndarray, pd.Series, ps.Column]:
    """Returns the string(s) with the indices marking the order of the string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    decreasing: bool, default is False
        If False, sorts from lowest to highest; if True sorts from highest to lowest.
    na_last: bool, default is True
        Where should NA go? True at the end, False at the beginning, None dropped.
    numeric: bool, default is False
        If True, will sort digits numerically, instead of as strings.

    Returns
    -------
    The indexes of our string in the order we desire
    """
    max_str_length = max([len(str(val)) for val in string]) + 1
    if isinstance(string, str):
        return 0
    elif isinstance(string, (list, tuple)):
        if na_last is None:
            string = [val for val in string if val not in [np.nan, None]]
        if numeric:
            if na_last is True:
                # For Python string comparisons, 9 is considered the highest for numbers
                string = [val if val not in [np.nan, None] else "9" * max_str_length for val in string]
            elif na_last is False:
                # Similarly, for Python string comparisons, 0 is considered the lowest for numbers
                string = [val if val not in [np.nan, None] else "0" for val in string]
            sorted_strings = index_natsorted(string)
        else:
            if na_last is True:
                # For Python string comparisons, 'z' is considered the highest for letters
                string = [val if val not in [np.nan, None] else "z" * max_str_length for val in string]
            elif na_last is False:
                # For Python string comparisons, 'A' is considered the lowest for letters
                string = [val if val not in [np.nan, None] else "A" for val in string]
            sorted_strings = sorted(range(len(string)), key=string.__getitem__)
    elif isinstance(string, (np.ndarray, np.generic)):
        if na_last is None:
            string = string[~np.isnan(string)]
            string = string[string != None]
        if numeric:
            if na_last is True:
                string = np.where(np.isin(string, [np.nan, None]), "9" * max_str_length, string)
            elif na_last is False:
                string = np.where(np.isin(string, [np.nan, None]), "0", string)
            sorted_strings = np.array(index_natsorted(string))
        else:
            if na_last is True:
                string = np.where(np.isin(string, [np.nan, None]), "z" * max_str_length, string)
            elif na_last is False:
                string = np.where(np.isin(string, [np.nan, None]), "A", string)
            sorted_strings = np.argsort(string)
    elif isinstance(string, pd.Series):
        if na_last is None:
            string = string.dropna()
        if numeric:
            if na_last is True:
                string = string.fillna("9" * max_str_length)
            elif na_last is False:
                string = string.fillna("0")
            sorted_strings = pd.Series(index_natsorted(string))
        else:
            if na_last is True:
                string = string.fillna("z" * max_str_length)
            elif na_last is False:
                string = string.fillna("A")
            sorted_strings = string.argsort()
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to order string")
    if decreasing:
        sorted_strings = sorted_strings[::-1]
    return sorted_strings


def str_sort(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    decreasing: bool = False,
    na_last: bool = True,
    numeric: bool = False,
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Sorts our strings based off the results of str_order

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    decreasing: bool, default is False
        If False, sorts from lowest to highest; if True sorts from highest to lowest.
    na_last: bool, default is True
        Where should NA go? True at the end, False at the beginning, None dropped.
    numeric: bool, default is False
        If True, will sort digits numerically, instead of as strings.

    Returns
    -------
    Our sorted string
    """
    if isinstance(string, str):
        return string
    else:
        indices = str_order(string, decreasing, na_last, numeric)
        if isinstance(string, (list, tuple)):
            return list((map(string.__getitem__, indices)))
        elif isinstance(string, (np.ndarray, np.generic, pd.Series)):
            return string[indices]
        elif isinstance(ps.Column):
            ...


def str_equal(
    x: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    y: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    ignore_case: bool = False,
) -> Union[bool, Sequence[bool], np.ndarray, pd.Series, ps.Column]:
    """Determines if two strings are equivalent

    Parameters
    ----------
    x, y: str or list/tuple or numpy array or pandas Series or pyspark column
        A pair of character vectors
    ignore_case: bool, default=False
        Ignore case when comparing strings?

    Returns
    -------
    Whether or not the strings are equal
    """
    if type(x) != type(y):
        raise TypeError("x and y must be of the same type")
    if isinstance(x, str):
        if ignore_case:
            return unicodedata.normalize("NFC", x.casefold()) == unicodedata.normalize("NFC", y.casefold())
        else:
            return unicodedata.normalize("NFC", x) == unicodedata.normalize("NFC", y)
    elif isinstance(x, pd.Series):
        if ignore_case:
            return pd.Series(
                np.where(
                    x.str.casefold().apply(lambda x: unicodedata.normalize("NFC", x))
                    == y.str.casefold().apply(lambda y: unicodedata.normalize("NFC", y)),
                    True,
                    False,
                )
            )
        else:
            return pd.Series(
                np.where(
                    x.apply(lambda x: unicodedata.normalize("NFC", x))
                    == y.apply(lambda y: unicodedata.normalize("NFC", y)),
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
    elif isinstance(x, (np.ndarray, np.generic)):
        if ignore_case:
            return np.compare_chararrays(
                np.fromiter(
                    (unicodedata.normalize("NFC", xi) for xi in np.char.upper(x)),
                    dtype=x.dtype,
                ),
                np.fromiter(
                    (unicodedata.normalize("NFC", yi) for yi in np.char.upper(y)),
                    dtype=y.dtype,
                ),
                "==",
                True,
            )
        else:
            return np.compare_chararrays(
                np.fromiter((unicodedata.normalize("NFC", xi) for xi in x), dtype=x.dtype),
                np.fromiter((unicodedata.normalize("NFC", yi) for yi in y), dtype=y.dtype),
                "==",
                True,
            )
    else:
        ...


# Whitespace Manipulation


def str_pad(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    width: Union[int, Sequence[int], np.ndarray, pd.Series, ps.Column],
    side: str = "right",
    pad: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column] = " ",
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Pad a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    width: int or list/tuple/numpy array/pandas Series
        Minimum width of padded strings
    side: str, default is right
        Side on which padding character is added (left, right or both).
    pad: str or list/tuple/numpy array/pandas Series, default is " "
        Single padding character (default is a space).

    Returns
    -------
    Our padded string
    """
    side = side.casefold()
    if side not in ["right", "left", "center"]:
        raise ValueError("Cannot determine where to pad string")
    if isinstance(string, str) and isinstance(width, int) and isinstance(pad, str):
        if side == "left":
            padded_string = string.rjust(width, pad)
        elif side == "right":
            padded_string = string.ljust(width, pad)
        else:
            padded_string = string.center(width, pad)
    else:
        if isinstance(string, pd.Series) and isinstance(pad, str) and isinstance(width, int):
            if side == "left":
                padded_string = string.str.rjust(width, pad)
            elif side == "center":
                padded_string = string.str.ljust(width, pad)
            else:
                padded_string = string.str.center(width, pad)
        elif isinstance(string, (np.ndarray, np.generic)):
            if side == "left":
                padded_string = np.char.rjust(string, width, pad)
            elif side == "right":
                padded_string = np.char.ljust(string, width, pad)
            else:
                padded_string = np.char.center(string, width, pad)
        else:
            if isinstance(string, str):
                if not isinstance(width, int):
                    string = [string] * len(width)
                else:
                    string = [string] * len(pad)
            if isinstance(width, int):
                if isinstance(pad, str):
                    if side == "left":
                        padded_string = [s.rjust(width, pad) for s in string]
                    elif side == "right":
                        padded_string = [s.ljust(width, pad) for s in string]
                    else:
                        padded_string = [s.center(width, pad) for s in string]
                else:
                    if side == "left":
                        padded_string = [string[i].rjust(width, pad[i]) for i in range(len(pad))]
                    elif side == "right":
                        padded_string = [string[i].ljust(width, pad[i]) for i in range(len(pad))]
                    else:
                        padded_string = [string[i].center(width, pad[i]) for i in range(len(pad))]
            else:
                if isinstance(pad, str):
                    if side == "left":
                        padded_string = [string[i].rjust(width[i], pad) for i in range(len(width))]
                    elif side == "right":
                        padded_string = [string[i].ljust(width[i], pad) for i in range(len(width))]
                    else:
                        padded_string = [string[i].center(width[i], pad) for i in range(len(width))]
                else:
                    if side == "left":
                        padded_string = [string[i].rjust(width[i], pad[i]) for i in range(len(width))]
                    elif side == "right":
                        padded_string = [string[i].ljust(width[i], pad[i]) for i in range(len(width))]
                    else:
                        padded_string = [string[i].center(width[i], pad[i]) for i in range(len(width))]
            if isinstance(string, (np.ndarray, np.generic)):
                # Testing it out, it turns out that it is significantly less efficient to initialize an array and
                # then iteratively re-populate it than it is to call np.array on list comprehension, like an order of
                # magnitude slower.
                # Additionally, one unique issue with numpy is that by default it only adds whitespace when you call
                # np.char.center(), which is super inefficient when you're calling it thousands, if not millions of times
                # So list comprehension makes more sense.
                padded_string = np.array(padded_string)
            elif isinstance(string, pd.Series):
                # This similarly applies to pandas as well.
                padded_string = pd.Series(padded_string)
    return padded_string


def str_trim(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    side: str = "both",
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Removes leading and trailing whitespace in our string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    side: str, {left, right, both}, default is both
        Side on which to remove whitespace.

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
        elif side == "right":
            return string.lstrip()
        else:
            return string.rstrip()
    elif isinstance(string, (list, tuple)):
        if side == "both":
            return [s.strip() for s in string]
        elif side == "right":
            return [s.lstrip() for s in string]
        else:
            return [s.rstrip() for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        if side == "both":
            return np.char.strip(string)
        elif side == "right":
            return np.char.lstrip(string)
        else:
            return np.char.rstrip(string)
    elif isinstance(string, pd.Series):
        if side == "both":
            return string.str.strip()
        elif side == "right":
            return string.str.lstrip()
        else:
            return string.str.rstrip()
    elif isinstance(string, ps.Column):
        ...
    else:
        raise ValueError("Cannot determine method for squishing strings")


def str_squish(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Squishes a string by removing leading and trailing whitespace as well as repeat whitespace inside a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.

    Returns
    -------
    Our squished string
    """
    if isinstance(string, str):
        squished_string = " ".join(string.split())
    elif isinstance(string, (list, tuple)):
        squished_string = [" ".join(s.split()) for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        # Using np.char.split() results in splits for every whitespace for every character, meaning we would have to
        # rejoin them together, which is less efficient than using calling np.array() on list comprehension.
        squished_string = np.array([" ".join(s.split()) for s in string])
    elif isinstance(string, pd.Series):
        # Similar to numpy, using df.str.split() results in splits for every whitespace for every character, meaning we
        # would have to rejoin them together, which is less efficient overall.
        squished_string = pd.Series([" ".join(s.split()) for s in string])
    elif isinstance(string, ps.Column):
        ...
    else:
        raise ValueError("Cannot determine method of squishing strings")
    return squished_string


# String Pattern Matching


def str_detect(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str,
    negate: bool = False,
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Determines if each string contains the regular expression

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    negate: bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Whether our pattern could be found within the string
    """
    if isinstance(string, str):
        if negate:
            match = re.search(pattern, string) is None
        else:
            match = re.search(pattern, string) is not None
    elif isinstance(string, (list, tuple)):
        if negate:
            match = [re.search(pattern, s) is None for s in string]
        else:
            match = [re.search(pattern, s) is not None for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
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
        ...
    else:
        raise TypeError("Cannot determine how to do string detection")
    return match


def str_count(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: Union[str, Sequence],
) -> Union[int, Sequence[int], np.ndarray, pd.Series, ps.Column]:
    """Count instances of pattern occurring within string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str or list
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").

    Returns
    -------
    The number of non-overlapping instances our pattern was found within the string
    """
    if isinstance(string, str):
        string_counts = len(re.findall(pattern, string))
    elif isinstance(string, (list, tuple)):
        if isinstance(pattern, str):
            string_counts = [len(re.findall(pattern, s)) for s in string]
        else:
            string_counts = [len(re.findall(pattern[i], string[i])) for i in range(len(pattern))]
    elif isinstance(string, (np.ndarray, np.generic)):
        if isinstance(pattern, str):
            string_counts = np.char.count(string, pattern)
        else:
            string_counts = np.array([len(re.findall(pattern[i], string[i])) for i in range(len(pattern))])
    elif isinstance(string, pd.Series):
        if isinstance(pattern, str):
            string_counts = string.str.count(pattern)
        else:
            string_counts = pd.Series([0] * len(pattern))
            string.index = np.arange(len(string))
            for i in range(len(pattern)):
                string_counts[i] = len(re.findall(pattern[i], string[i]))
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to do string count")
    return string_counts


def str_subset(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str,
    negate: bool = False,
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Filters our string based on str_detect

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    negate: bool, default is False
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
    elif isinstance(string, ps.Column):
        ...
    else:
        return string[str_detect(string, pattern, negate)]


def str_which(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str,
    negate: bool = False,
) -> Optional[Union[int, Sequence[int], np.ndarray, pd.Series, ps.Column]]:
    """Filters our string based on str_detect

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    negate: bool, default is False
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
    elif isinstance(string, (np.ndarray, np.generic)):
        return np.where(string_loc)[0]
    elif isinstance(string, pd.Series):
        return pd.Series(string[string_loc.isin([True])].index)
    elif isinstance(string, ps.Column):
        ...
    else:
        raise ValueError("Cannot determine how to do string which")


def _str_replace(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: Union[str, Sequence[str]],
    replacement: Union[str, Sequence[str]],
    how: str = "all",
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Replaces either the first or all instance(s) of pattern in string with replacement

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str or list
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    replacement: str or list
        A character vector of replacements. Should be either length one, or the same length as string or pattern.
        References of the form \1, \2, etc will be replaced with the contents of the respective matched group (created by ()).
        To perform multiple replacements in each element of string, pass a named vector (c(pattern1 = replacement1)) to
        str_replace_all. Alternatively, pass a function to replacement: it will be called once for each match and its
        return value will be used to replace the match. To replace the complete string with NA, use
        replacement = None.
    how: str, default is 'all'
        Whether we are replacing all matches or only the first

    Returns
    -------
    The string, but with either the first or all matched patterns removed
    """
    if how.casefold() == "all":
        count = 0
    else:
        count = 1
    if isinstance(string, str):
        if replacement is None or replacement is np.nan:
            return replacement if pattern in string else string
        else:
            return re.sub(pattern, replacement, string, count=count)
    elif isinstance(string, (list, tuple, np.ndarray, np.generic)):
        if isinstance(replacement, str):
            if isinstance(string, (list, tuple)):
                return [re.sub(pattern, replacement, s, count=count) for s in string]
            else:
                return np.array(
                    list(
                        map(
                            lambda v: re.sub(pattern, replacement, v, count=count),
                            string,
                        )
                    )
                )
        elif replacement is None or replacement is np.nan:
            if isinstance(string, (list, tuple)):
                return [replacement if pattern in s else s for s in string]
            else:
                return np.array(list(map(lambda v: replacement if pattern in v else v, string)))
        else:
            if isinstance(pattern, str):
                match = [re.sub(pattern, replacement[i], string[i], count=count) for i in range(len(string))]
            else:
                match = [re.sub(pattern[i], replacement[i], string[i], count=count) for i in range(len(string))]
            if isinstance(string, (np.ndarray, np.generic)):
                match = np.array(match)
            return match
    elif isinstance(string, pd.Series):
        if isinstance(replacement, str) or replacement is None or replacement is np.nan:
            return string.str.replace(pattern, replacement, regex=True, n=-1)
        else:
            replacement_series = pd.Series([""] * len(string))
            replacement_series.index = np.arange(len(string))
            for i in range(len(pattern)):
                if isinstance(pattern, str):
                    replacement_series[i] = re.sub(pattern, replacement[i], string[i], count=count)
                else:
                    replacement_series[i] = re.sub(pattern[i], replacement[i], string[i], count=count)
            return replacement_series
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to do string replace")


def str_replace(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: Union[str, Sequence[str]],
    replacement: Union[str, Sequence[str]],
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Replaces the first instance of pattern in string with replacement"""
    return _str_replace(string, pattern, replacement, how="first")


def str_replace_all(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: Union[str, Sequence[str]],
    replacement: Union[str, Sequence[str]],
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Replaces all instances of pattern in string with replacement"""
    return _str_replace(string, pattern, replacement, how="all")


def str_remove(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column], pattern: str
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Remove the first pattern in our string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
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
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column], pattern: str
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Remove all patterns in our string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
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


def str_split(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str = " ",
    n: int = -1,
    simplify: bool = False,
) -> Union[Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Split a string into pieces

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str, default is " "
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    n: int, default is -1
        Number of pieces to return. Default -1 uses all possible split positions.
    simplify: bool, default is False
        If False, the default, returns a list of character vectors. If True, returns a character matrix.

    Returns
    -------
    Our string, but split along the pattern
    """
    if isinstance(string, str):
        return string.split(pattern, maxsplit=n)
    elif isinstance(string, (list, tuple)):
        split_string = [s.split(pattern, maxsplit=n) for s in string]
        if simplify:
            length = max(map(len, split_string))
            split_string = [xi + [""] * (length - len(xi)) for xi in split_string]
        return split_string
    elif isinstance(string, (np.ndarray, np.generic)):
        split_string = np.char.split(string, pattern, maxsplit=n)
        if simplify:
            length = max(map(len, split_string))
            split_string = np.array([np.array(xi + [""] * (length - len(xi))) for xi in split_string])
        return split_string
    elif isinstance(string, pd.Series):
        split_string = string.str.split(pattern, n=n, expand=simplify)
        if simplify:
            split_string = split_string.fillna("")
        return split_string
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to do string splitting")


def str_split_fixed(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str = " ",
    n: int = -1,
) -> Union[Sequence[str], np.ndarray, pd.DataFrame, ps.DataFrame]:
    """Returns a character matrix. Essentially a wrapper for str_split with simplify=True

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str, default is " "
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    n: int, default is -1
        For str_split_fixed, if n is greater than the number of pieces, the result will be padded with empty strings.

    Returns
    -------
    A matrix of character splits
    """
    return str_split(string, pattern, n, simplify=True)


def str_split_n(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str = " ",
    n: int = 0,
) -> Optional[Union[Sequence[str], np.ndarray, pd.Series, ps.Column]]:
    """Returns the nth index of a split string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str, default is " "
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    n: int, default is 0
        For str_split_n, n is the desired index of each element of the split string. When there are fewer pieces than n,
        return np.nan.

    Returns
    -------
    A character vector containing the nth index of each string split.
    """
    split_string = str_split(string, pattern, n=-1, simplify=True)
    if isinstance(string, str):
        if n >= len(split_string):
            return None
        else:
            return split_string[n]
    elif isinstance(string, (list, tuple)):
        length = len(split_string[0])
        if n >= length:
            raise ValueError("Index greater than number of elements")
        return [el[n] if el[n] != "" else None for el in split_string]
    elif isinstance(string, (np.ndarray, np.generic)):
        length = len(split_string[0])
        if n >= length:
            raise ValueError("Index greater than number of elements")
        split_string = split_string.flatten()
        return pd.Series(split_string[n::length]).replace("", np.nan).to_numpy()
    elif isinstance(string, (pd.Series, pd.DataFrame)):
        if n >= len(split_string.columns):
            raise ValueError("Index greater than number of elements")
        return split_string.iloc[:, n].replace("", np.nan)
    elif isinstance(string, (ps.Column, ps.DataFrame)):
        ...
    else:
        raise TypeError("Cannot determine how to do string splitting for fixed n")


def str_starts(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str,
    negate: bool = False,
) -> Union[bool, Sequence[bool], np.ndarray, pd.Series, ps.Column]:
    """Detect the presence or absence of a pattern at the beginning of a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    negate: bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Either a bool or a container of bools indicating whether we can find our pattern at the start of the string
    """
    pattern = "^" + pattern
    return str_detect(string, pattern, negate)


def str_ends(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str,
    negate: bool = False,
) -> Union[bool, Sequence[bool], np.ndarray, pd.Series, ps.Column]:
    """Detect the presence or absence of a pattern at the end of a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    negate: bool, default is False
        If True, return non-matching elements.

    Returns
    -------
    Either a bool or a container of bools indicating whether we can find our pattern at the end of the string
    """
    pattern = pattern + "$"
    return str_detect(string, pattern, negate)


def str_extract(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column], pattern: str
) -> Optional[Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]]:
    """Extract first matching patterns from a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").

    Returns
    -------
    First match within each string
    """
    if isinstance(string, str):
        if re.search(pattern, string) is not None:
            return re.search(pattern, string).group(0)
        else:
            return None
    elif isinstance(string, (list, tuple, np.ndarray, np.generic)):
        extract = [
            re.search(pattern, s).group(0) if s not in [None, np.nan] and re.search(pattern, s) is not None else None
            for s in string
        ]
        if isinstance(string, (np.ndarray, np.generic)):
            extract = np.array(extract)
        return extract
    elif isinstance(string, pd.Series):
        return pd.Series(
            [
                re.search(pattern, s[1]).group(0)
                if s[1] not in [None, np.nan] and re.search(pattern, s[1]) is not None
                else None
                for s in string.iteritems()
            ]
        )
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine method of string extraction")


def str_extract_all(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column],
    pattern: str,
    simplify: bool = False,
) -> Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column]:
    """Extract all matching patterns from a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").
    simplify: bool, default is True
        If False, the default, returns a list of character vectors. If True returns a character matrix.

    Returns
    -------
    All non-overlapping matches within each string
    """
    if isinstance(string, str):
        match = re.findall(pattern, string)
    elif isinstance(string, (list, tuple, np.ndarray, np.generic)):
        match = [re.findall(pattern, s) for s in string]
        if simplify:
            length = max(map(len, match))
            match = [xi + [""] * (length - len(xi)) for xi in match]
            match = ["" if x is None else x for x in match]
        if isinstance(string, (np.ndarray, np.generic)):
            match = np.array(match, dtype="object")
    elif isinstance(string, pd.Series):
        if "(" not in pattern:
            pattern = "(" + pattern + ")"
        match = string.str.extractall(pattern)
        if simplify:
            if len(match.index) > 2:
                if len(match.columns) == 1:
                    match = pd.pivot(match.reset_index(), index="level_0", columns="match", values=0)
                else:
                    value_cols = match.columns.difference(["level_0", "match"])
                    match = pd.pivot(
                        match.reset_index(),
                        index="level_0",
                        columns="match",
                        values=value_cols,
                    )
                match = match.fillna("")
            else:
                flat_match = pd.Series([""] * len(string), name="match")
                flat_match[flat_match.index.isin(match.reset_index().level_0)] = (
                    match.rename({0: "val"}, axis=1).reset_index()["val"].values
                )
                match = flat_match
    elif isinstance(string, ps.Column):
        ...
    return match


def str_match(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column], pattern: str
) -> Optional[Union[Sequence[str], np.ndarray, pd.DataFrame, ps.DataFrame]]:
    """Extract first matched group from a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").

    Returns
    -------
    A container with our first matched groups, and None/np.nan if no match detected
    """
    whole_match = str_extract(string, pattern)
    if isinstance(string, str):
        if whole_match is None:
            return None
        else:
            partial_match = list(map(list, str_extract_all(whole_match, pattern, simplify=True)))[0]
            return_match = [whole_match] + partial_match
    elif isinstance(string, (list, tuple, np.ndarray, np.generic)):
        if isinstance(string, (np.ndarray, np.generic)):
            whole_match[whole_match == None] = ""
        else:
            whole_match = [s if s is not None else "" for s in whole_match]
        partial_match = str_extract_all(whole_match, pattern, simplify=True)
        return_match = [[a] + [elem for elem in b[0]] for a, b in zip(whole_match, partial_match)]
        max_length = max(len(x) for x in return_match)
        return_match = [
            val if len(val) == max_length else [None] * max_length for index, val in enumerate(return_match)
        ]
        if isinstance(string, (np.ndarray, np.generic)):
            return_match = np.array(return_match)
    elif isinstance(string, pd.Series):
        whole_match = whole_match.rename("whole_match")
        partial_match = str_extract_all(whole_match, pattern, simplify=True)
        return_match = pd.merge(whole_match, partial_match, how="left", left_index=True, right_index=True)
        return_match = return_match.replace("", np.nan)
    elif isinstance(string, ps.Column):
        ...
    return return_match


def str_match_all(
    string: Union[str, Sequence[str], np.ndarray, pd.Series, ps.Column], pattern: str
) -> Optional[Union[Sequence[str], np.ndarray, pd.DataFrame, ps.DataFrame]]:
    """Extract all matched groups from a string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark column
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").

    Returns
    -------
    A container with all our matched groups, and None/np.nan if no match detected
    """
    if isinstance(string, ps.Column):
        ...
    else:
        # One issue we have is that str_extract only finds the first total instance of a capture group in our string,
        # while str_extract_all finds all the inner capture groups within our string. This works for str_match, where
        # we need the first total match and all its inner matches, but doesn't work for str_match_all, where we need all
        # total matches. So we need to iteratively go through the string, save all total matches and then get all
        # inner matches
        whole_match = []
        if isinstance(string, str):
            if string not in [np.nan, None] and re.search(pattern, string) is not None:
                whole_match = []
                for match in re.finditer(pattern, string):
                    whole_match.append(match.group(0))
            else:
                return ""
        else:
            for s in string:
                if s not in [np.nan, None] and re.search(pattern, s) is not None:
                    for match in re.finditer(pattern, s):
                        whole_match.append(match.group(0))
                else:
                    whole_match.append("")
        if isinstance(string, pd.Series):
            whole_match = pd.Series(whole_match, name="whole_match")
            partial_match = str_extract_all(whole_match, pattern).reset_index().drop(["match"], axis=1)
            return_match = (
                pd.merge(
                    whole_match,
                    partial_match,
                    left_index=True,
                    right_on="level_0",
                    how="left",
                )
                .drop(["level_0"], axis=1)
                .fillna("")
            )
            return_match.index = np.arange(len(return_match))
        else:
            partial_match = str_extract_all(whole_match, pattern, simplify=True)
            return_match = [[a] + [elem for elem in b[0]] for a, b in zip(whole_match, partial_match)]
            max_length = max(len(x) for x in return_match)
            return_match = [
                val if len(val) == max_length else [""] * max_length for index, val in enumerate(return_match)
            ]
            if isinstance(string, (np.ndarray, np.generic)):
                return_match = np.array(return_match)
    return return_match


# Combining Strings


def str_c(
    strings: Union[Sequence[str], np.ndarray, pd.Series, ps.Column],
    sep: str = "",
    collapse: Optional[str] = None,
) -> str:
    """

    Parameters
    ----------
    strings: list of lists or array of arrays or pandas DataFrame
        One or more character vectors
    sep: str, default=""
        String to insert between input vectors
    collapse: str, default=None
        Optional string used to combine outputs into a single string

    Returns
    -------
    If `collapse=None`, a character vector with length equal to the longest input. Else, a
    character vector of length 1.
    """
    if isinstance(strings, (list, tuple)):
        max_length = max([len(s) for s in strings])
        return_string = [""] * max_length
        for index, string in enumerate(strings):
            if len(string) < max_length:
                strings[index] = string * (max_length // len(string)) + string[0 : max_length % len(string)]
        for row, ss in enumerate(strings[0]):
            for col in range(1, len(strings)):
                ss += sep + strings[col][row]
            return_string[row] = ss
    elif isinstance(strings, (np.generic, np.ndarray)):
        max_length = max(map(len, strings))
        return_string = np.repeat("", max_length)
        ...
    elif isinstance(strings, pd.DataFrame):
        ...
    elif isinstance(strings, ps.DataFrame):
        ...
    else:
        raise TypeError("Cannot identify character vector")
    if collapse is not None:
        return collapse.join(return_string)
    return return_string


def _remove_accents(input_str: str) -> str:
    """Removes all accents from a string

    Parameters
    ----------
    input_str: string
        The string we wish to remove accents from

    Returns
    -------
    Our string, but with all accents removed
    """
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
