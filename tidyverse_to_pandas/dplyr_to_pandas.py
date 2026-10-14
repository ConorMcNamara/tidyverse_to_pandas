"""The goal of this is to convert dplyr syntax to pandas."""

import re

import pandas as pd

from tidyverse_to_pandas import rebase_dplyr_to_pandas as _rebase
from tidyverse_to_pandas.rebase_dplyr_to_pandas import _eval_expression, _split_assignment

# Transform Variables


def mutate(data, *args):
    """Convert a string function into an assignment for pandas dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        The data frame for which we are creating new variables using mutate()
    *args : str
        The variable functions, for example "new_col = col_1 / col_2"

    Returns
    -------
    data : pandas DataFrame
        The data frame, with the new columns created using mutate

    For example, suppose we had a dataframe like
    a b
    1 2
    3 4
    5 6

    Then running mutate(df, "c = a + b") will return
    a b c
    1 2 3
    3 4 7
    5 6 11
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use mutate on a non-DataFrame")
    return _rebase.mutate(data, list(args))


def transmute(data, *args):
    """Convert a string function into an assignment for pandas DataFrame.

    Parameters
    ----------
    data : pandas DataFrame
        The data frame for which we are creating new variables using transmute()
    *args : str
        The variable functions, for example "new_col = col_1 / col_2"

    Returns
    -------
    data : pandas DataFrame
        The data frame, with only the new columns created using transmute

    For example, suppose we had a dataframe like
    a b
    1 2
    3 4
    5 6

    Then running transmute(df, "c = a + b") will return
    c
    3
    7
    11
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use transmute on a non-DataFrame")
    return _rebase.transmute(data, list(args))


# Select/Rename Variables


def rename(data, *args):
    """Rename columns using the 'x = x_1' arguments, and keeps all the available columns.

    Returns
    -------
    data : pandas DataFrame
        The DataFrame, but with the columns renamed

    For example, suppose we had a dataframe like
    a b
    1 2
    3 4
    5 6

    Then running rename(df, "a = col_a") will return
    col_a b
    1     2
    3     4
    5     6

    Parameter
    ---------
    data: pandas DataFrame
        The data frame for which we are renaming current columns
    *args: str
        The columns we are renaming, for example, "x = x_1"
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use rename on a non-DataFrame")
    return _rebase.rename(data, list(args))


def starts_with(column_name, match, ignore_case=True):
    """Test whether ``column_name`` starts with ``match``.

    Parameters
    ----------
    column_name : str
        The column name being tested against the ``match`` pattern
    match : str
        The substring or pattern searched for at the start of ``column_name``
    ignore_case : bool, default is True
        If True, the comparison is performed case-insensitively

    Returns
    -------
    re.Match or None
        The match object if ``column_name`` starts with ``match``, otherwise None
    """
    # As in tidyselect, the match is a literal string rather than a regular expression
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(rf"^{re.escape(match)}", column_name, flags)


def ends_with(column_name, match, ignore_case=True):
    """Test whether ``column_name`` ends with ``match``.

    Parameters
    ----------
    column_name : str
        The column name being tested against the ``match`` pattern
    match : str
        The substring or pattern searched for at the end of ``column_name``
    ignore_case : bool, default is True
        If True, the comparison is performed case-insensitively

    Returns
    -------
    re.Match or None
        The match object if ``column_name`` ends with ``match``, otherwise None
    """
    # As in tidyselect, the match is a literal string rather than a regular expression
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(rf"{re.escape(match)}$", column_name, flags)


def contains(column_name, match, ignore_case=True):
    """Test whether ``column_name`` contains ``match``.

    Parameters
    ----------
    column_name : str
        The column name being tested against the ``match`` pattern
    match : str
        The substring or pattern searched for anywhere within ``column_name``
    ignore_case : bool, default is True
        If True, the comparison is performed case-insensitively

    Returns
    -------
    re.Match or None
        The match object if ``column_name`` contains ``match``, otherwise None
    """
    # As in tidyselect, the match is a literal string rather than a regular expression
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(re.escape(match), column_name, flags)


def select(data, *args):
    """Select columns based on criteria in args.

    Returns
    -------
    new_data : pandas DataFrame
        A DataFrame with only the selected columns

    For example, suppose we had a dataframe like
    a b
    1 2
    3 4
    5 6

    Then running select(df, "a") will return
    a
    1
    3
    5

    Parameter
    ---------
    data: pandas DataFrame
        The data frame for which we are renaming current columns
    *args: str
        The columns we are selecting, for example, "x" or "x_1"
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use select on a non-DataFrame")
    columns = list(data.columns)
    selected: list = []
    # Selections are applied in order. A leading exclusion starts from every column, as in tidyselect.
    for i, arg in enumerate(args):
        arg = arg.strip()
        exclude = arg.startswith("-") and arg not in columns
        matched = _match_selection(data, arg[1:].strip() if exclude else arg)
        if exclude:
            if i == 0:
                selected = list(columns)
            selected = [col for col in selected if col not in matched]
        else:
            selected += [col for col in matched if col not in selected]
    return data[selected]


def _strip_quotes(text):
    return text.strip().strip("'\"")


def _match_selection(data, arg):
    """Return the columns of ``data`` matched by a single (non-negated) select argument."""
    columns = list(data.columns)
    if arg in columns:
        return [arg]
    helper = re.fullmatch(r"(starts_with|ends_with|contains)\((.*)\)", arg)
    if helper:
        func = {"starts_with": starts_with, "ends_with": ends_with, "contains": contains}[helper.group(1)]
        match = _strip_quotes(helper.group(2))
        return [col for col in columns if func(col, match)]
    if re.fullmatch(r"everything\(\s*\)", arg):
        return columns
    num_range = re.fullmatch(r"num_range\((.+?),\s*(\d+)\s*:\s*(\d+)\s*\)", arg)
    if num_range:
        prefix = _strip_quotes(num_range.group(1))
        start, end = int(num_range.group(2)), int(num_range.group(3))
        step = 1 if end >= start else -1
        return [f"{prefix}{i}" for i in range(start, end + step, step) if f"{prefix}{i}" in columns]
    last = re.fullmatch(r"last_col\((?:\s*(?:offset\s*=\s*)?(\d+))?\s*\)", arg)
    if last:
        offset = int(last.group(1) or 0)
        if offset >= len(columns):
            raise Exception("Number of offsets is greater than number of columns")
        return [columns[-(offset + 1)]]
    if ":" in arg:
        start_col, end_col = (part.strip() for part in arg.split(":", 1))
        if start_col in columns and end_col in columns:
            return columns[columns.index(start_col) : columns.index(end_col) + 1]
    raise Exception(f"Selected column {arg} not found in data frame")


# Filter Data


def filter(data, *args):
    """Filter data based on arguments from args.

    Parameters
    ----------
    data : pandas DataFrame
        The dataframe for which we are filtering the data on
    *args : str
        The filter conditions we are applying on our dataframe

    Returns
    -------
    filtered_data : pandas DataFrame
        The dataframe, after we've applied all filtering conditions

    For example, suppose we had a dataframe like
    a b
    1 2
    3 4
    5 6

    Then running filter(df, "a >= median(a)") will return
    a
    3
    5
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use filter on non-DataFrame")
    return _rebase.filter(data, list(args))


# Summarise Data


def summarise(data, *args):
    """Summarises data based on arguments from args.

    Parameters
    ----------
    data : pandas DataFrame
        The dataframe for which we are trying to summarise the data on
    *args : str
        The columns we are performing summary calculations on, as well as type of summary calculations

    Returns
    -------
    summarised_data : pandas DataFrame
        The dataframe containing our summarised results

    For example, suppose we had a dataframe like
    a b
    1 2
    3 4
    5 6

    Then running summarise(df, "mean(a)") will return
    4.5
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use summarise on a non-DataFrame")
    summarised = {}
    for arg in args:
        try:
            name, expression = _split_assignment(arg)
        except ValueError:
            name, expression = arg, arg
        value, _ = _eval_expression(data, expression)
        summarised[name] = [value]
    return pd.DataFrame(summarised)


# Sort data


def arrange(data, *args):
    """Sorts the data based on columns provided, as well as if they're in 'desc()'.

    Parameters
    ----------
    data : pandas DataFrame
        The dataframe for which we are trying to sort the data on
    *args : str
        The columns we are sorting the dataframe on

    Returns
    -------
    sorted_data : pandas DataFrame
        The sorted data based on arguments provided in *args

    For example, suppose we had a dataframe like
    a b
    1 2
    3 4
    5 6

    Then running arrange(df, desc(a)) will return
    a b
    5 6
    3 4
    1 2
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use arrange on a non-DataFrame")
    return _rebase.arrange(data, list(args))
