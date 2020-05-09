import pandas as pd
import numpy as np
import pyspark.sql as ps
import re


# Character Manipulation

def str_length(string):
    """Calculates the length of each string

    Parameters
    ----------
    string: str or list or numpy array or pandas Series or pyspark DataFrame
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
    elif isinstance(string, ps.DataFrame):
        ...
    else:
        raise TypeError("Cannot determine how to calculate string length")


def str_sub(string, start, end=None):
    """Extract substrings from a character vector

    One important note is that R and Python have vastly different indexing rules. R is [start, end] whereas Python is
    [start, end). Additionally, R has its indexes start at 1 and end at len(string), whereas Python has its indexes
    start at 0 and end at len(string) - 1. I figure that it would be too confusing to intermingle R indexing rules with
    Python, so this function follows Python indexing conventions.

    Parameters
    ----------
    string: str or list or numpy array or pandas Series or pyspark DataFrame
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
    elif isinstance(string, ps.DataFrame):
        ...
    else:
        raise TypeError("Cannot determine how to do string subset")


def str_dup(string, num_dupes):
    """Changes our string by duplicating it

    Parameters
    ----------
    string: str or list or numpy array or pandas Series or pyspark DataFrame
        Input vector. Either a character vector, or something coercible to one
    num_dupes: int or list/tuple or numpy array
        Number of duplications. An integer implies equal number of duplications while a list/tuple or array specifies
        the number of duplications per entry
    Returns
    -------

    """
    if isinstance(string, str):
        if not isinstance(num_dupes, int):
            raise TypeError("Cannot determine number of duplications using type {}. Use integer instead".format(type(num_dupes)))
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
            return np.array([''.join(row) for row in split_repeat])[:-1]
        else:
            return np.array([''.join(row) for row in split_repeat])
    elif isinstance(string, pd.Series):
        return string.str.repeat(repeats=num_dupes)
    elif isinstance(string, ps.DataFrame):
        ...
    else:
        raise TypeError("Cannot determine how to do string duplication")


# String Pattern Matching

def str_detect(string, pattern, negate=False):
    """Determines if each string contains the regular expression

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark dataframe
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
    elif isinstance(string, ps.DataFrame):
        ...
    else:
        raise TypeError("Cannot determine how to do string detection")
    return match


def str_count(string, pattern):
    """Count instances of pattern occurring within string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark dataframe
        Input vector. Either a character vector, or something coercible to one.
    pattern: str
        Pattern to look for. The default interpretation is a regular expression, as described in
        stringi::stringi-search-regex. Control options with regex(). Match a fixed string (i.e. by comparing only bytes),
        using fixed(). This is fast, but approximate. Generally, for matching human text, you'll want coll() which
        respects character matching rules for the specified locale. Match character, word, line and sentence boundaries
        with boundary(). An empty pattern, "", is equivalent to boundary("character").

    Returns
    -------

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
    elif isinstance(string, ps.DataFrame):
        ...
    else:
        raise TypeError("Cannot determine how to do string count")
    return string_counts


def str_subset(string, pattern, negate=True):
    """Filters our string based on str_detect

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark dataframe
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

    """
    if isinstance(string, str):
        if str_detect(string, pattern, negate):
            return string
        else:
            return ""
    else:
        return string[str_detect(string, pattern, negate)]

def str_which(string, pattern, negate):
    """Returns the index of the first index of the filtered string

    Parameters
    ----------
    string: str or list/tuple or numpy array or pandas Series or pyspark dataframe
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

    """
    filtered_string = str_subset(string, pattern, negate)
    if len(filtered_string) == 0:
        return filtered_string
    else:
        if isinstance(filtered_string, str):
            ...


def str_replace(string, pattern, replacement):
    """

    Parameters
    ----------
    string
    pattern
    replacement

    Returns
    -------

    """
    if isinstance(string, str):
        return re.sub(pattern, replacement, string)
    elif isinstance(string, (list, tuple)):
        return [re.sub(pattern, replacement, s) for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        ...
    elif isinstance(string, pd.Series):
        return string.str.replace(pattern, replacement, regex=True)
    elif isinstance(string, ps.DataFrame):
        return np.array(list(map(lambda v: re.sub(pattern, replacement, v), string)))
    else:
        raise TypeError("Cannot determine how to do string replace")