import pandas as pd
import numpy as np
import pyspark.sql as ps
import re
from itertools import compress
from string import capwords
from natsort import index_natsorted

# Character Manipulation


def str_length(string):
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


def str_sub(string, start, end=None):
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


def str_dup(string, num_dupes):
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
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to do string duplication")


def str_flatten(string, collapse=""):
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
        return '{}'.format(collapse).join([char for char in string])
    elif isinstance(string, (np.ndarray, np.generic)):
        # Ironically, we can call this for both pandas and lists, but I wanted to show the various ways of doing it
        # rather than simply calling the iterable.
        return '{}'.format(collapse).join(string)
    elif isinstance(string, pd.Series):
        return '{}'.format(collapse).join(string.values.flatten())


def str_trunc(string, width, side="right", ellipsis="..."):
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
    if side.casefold() not in ['right', 'left', 'center']:
        raise ValueError("Cannot determine location of ellipsis.")
    if isinstance(string, str):
        if side.casefold() == 'right':
            return string[:width] + str(ellipsis)
        elif side.casefold() == 'left':
            return str(ellipsis) + string[-width:]
        else:
            return string[:width//2] + ellipsis + string[-width//2:]
    elif isinstance(string, (list, tuple)):
        if side.casefold() == 'right':
            return [s[:width] + str(ellipsis) for s in string]
        elif side.casefold() == 'left':
            return [str(ellipsis) + s[-width:] for s in string]
        else:
            return [s[:width//2] + str(ellipsis) + s[-width//2:] for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        # The way np.frompyfunc works is that it takes in a function (such as a lambda expression) and then you feed it
        # the parameters needed to run it. Essentially, it's a way of converting f(x) to something that numpy can apply
        # to all elements of the array
        if side.casefold() == 'right':
            return np.frompyfunc(lambda s: s[:width] + str(ellipsis), 1, 1)(string)
        elif side.casefold() == 'left':
            return np.frompyfunc(lambda s: str(ellipsis) + s[-width:], 1, 1)(string)
        else:
            return np.frompyfunc(lambda s: s[:width//2] + str(ellipsis) + s[-width//2], 1, 1)(string)
    elif isinstance(string, pd.Series):
        if side.casefold() == 'right':
            return string.str.slice(stop=width) + str(ellipsis)
        elif side.casefold() == 'left':
            return string.str.slice(start=width) + str(ellipsis)
        else:
            return string.str.slice(stop=width//2) + str(ellipsis) + string.str.slice(start=width//2)
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to perform string truncation")


def str_replace_na(string, replacement="NA"):
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


# Case Transformations

def str_to_upper(string):
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


def str_to_title(string):
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


def str_to_lower(string, locale='en'):
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
        if locale == 'en':
            return string.casefold()
        else:
            return string.lower()
    elif isinstance(string, (list, tuple)):
        if locale == 'en':
            return [s.casefold() for s in string]
        else:
            return [s.lower() for s in string]
    elif isinstance(string, (np.ndarray, np.generic)):
        if locale == 'en':
            return np.array(list(map(lambda v: v.casefold(), string)))
        else:
            return np.char.lower(string)
    elif isinstance(string, pd.Series):
        if locale == 'en':
            return string.str.casefold()
        else:
            return string.str.lower()
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to lower strings")


def str_to_sentence(string):
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


# String Ordering
def str_order(string, decreasing=False, na_last=True, numeric=False):
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
                string = [val if val not in [np.nan, None] else '9' * max_str_length for val in string]
            elif na_last is False:
                # Similarly, for Python string comparisons, 0 is considered the lowest for numbers
                string = [val if val not in [np.nan, None] else '0' for val in string]
            sorted_strings = index_natsorted(string)
        else:
            if na_last is True:
                # For Python string comparisons, 'z' is considered the highest for letters
                string = [val if val not in [np.nan, None] else 'z' * max_str_length for val in string]
            elif na_last is False:
                # For Python string comparisons, 'a' is considered the lowest for letters
                string = [val if val not in [np.nan, None] else 'a' for val in string]
            sorted_strings = sorted(range(len(string)), key=string.__getitem__)
    elif isinstance(string, (np.ndarray, np.generic)):
        if na_last is None:
            string = string[~np.isnan(string)]
            string = string[string != None]
        if numeric:
            if na_last is True:
                string = np.where(np.isin(string, [np.nan, None]), '9' * max_str_length, string)
            elif na_last is False:
                string = np.where(np.isin(string, [np.nan, None]), '0', string)
            sorted_strings = np.array(index_natsorted(string))
        else:
            if na_last is True:
                string = np.where(np.isin(string, [np.nan, None]), 'z' * max_str_length, string)
            elif na_last is False:
                string = np.where(np.isin(string, [np.nan, None]), 'a', string)
            sorted_strings = np.argsort(string)
    elif isinstance(string, pd.Series):
        if na_last is None:
            string = string.dropna()
        if numeric:
            if na_last is True:
                string = string.fillna('9' * max_str_length)
            elif na_last is False:
                string = string.fillna('0')
            sorted_strings = pd.Series(index_natsorted(string))
        else:
            if na_last is True:
                string = string.fillna('z' * max_str_length)
            elif na_last is False:
                string = string.fillna('a')
            sorted_strings = string.argsort()
    elif isinstance(string, ps.Column):
        ...
    else:
        raise TypeError("Cannot determine how to order string")
    if decreasing:
        sorted_strings = sorted_strings[::-1]
    return sorted_strings


def str_sort(string, decreasing=False, na_last=True, numeric=False):
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


# String Pattern Matching

def str_detect(string, pattern, negate=False):
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


def str_count(string, pattern):
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


def str_subset(string, pattern, negate=False):
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

    """
    if isinstance(string, str):
        if str_detect(string, pattern, negate):
            return string
        else:
            return ""
    elif isinstance(string, list):
        return list(compress(string, str_detect(string, pattern, negate)))
    else:
        return string[str_detect(string, pattern, negate)]


def _str_replace(string, pattern, replacement, how='all'):
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
    if how.casefold() == 'all':
        count = 0
    else:
        count = 1
    if isinstance(string, str):
        if replacement is None or replacement is np.nan:
            return replacement if pattern in string else string
        else:
            return re.sub(pattern, replacement, string, count=count)
    elif isinstance(string, (list, tuple)):
        if isinstance(replacement, str):
            return [re.sub(pattern, replacement, s, count=count) for s in string]
        elif replacement is None or replacement is np.nan:
            return [replacement if pattern in s else s for s in string]
        else:
            if isinstance(pattern, str):
                return [re.sub(pattern, replacement[i], string[i], count=count) for i in range(len(string))]
            else:
                return [re.sub(pattern[i], replacement[i], string[i], count=count) for i in range(len(string))]
    elif isinstance(string, (np.ndarray, np.generic)):
        if isinstance(replacement, str):
            return np.array(list(map(lambda v: re.sub(pattern, replacement, v, count=count), string)))
        elif replacement is None or replacement is np.nan:
            return np.array(list(map(lambda v: replacement if pattern in v else v, string)))
        else:
            if isinstance(pattern, str):
                return np.array([re.sub(pattern, replacement[i], string[i], count=count) for i in range(len(string))])
            else:
                return np.array([re.sub(pattern[i], replacement[i], string[i], count=count) for i in range(len(string))])
    elif isinstance(string, pd.Series):
        if isinstance(replacement, str) or replacement is None or replacement is np.nan:
            return string.str.replace(pattern, replacement, regex=True, n=-1)
        else:
            replacement_series = pd.Series([''] * len(string))
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


def str_replace(string, pattern, replacement):
    """Replaces the first instance of pattern in string with replacement"""
    return _str_replace(string, pattern, replacement, how='first')


def str_replace_all(string, pattern, replacement):
    """Replaces all instances of pattern in string with replacement"""
    return _str_replace(string, pattern, replacement, how='all')


def str_remove(string, pattern):
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


def str_remove_all(string, pattern):
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


def str_split(string, pattern=" ", n=-1, simplify=False):
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
    elif isinstance(string, list):
        split_string = [s.split(pattern, maxsplit=n) for s in string]
        if simplify:
            length = max(map(len, split_string))
            split_string = [xi + [""] * (length-len(xi)) for xi in split_string]
        return split_string
    elif isinstance(string, (np.ndarray, np.generic)):
        split_string = np.char.split(string, pattern, maxsplit=n)
        if simplify:
            length = max(map(len, split_string))
            split_string = np.array([np.array(xi + [""] * (length-len(xi))) for xi in split_string])
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


def str_split_fixed(string, pattern=" ", n=-1):
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


def str_split_n(string, pattern=" ", n=0):
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
    elif isinstance(string, list):
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
