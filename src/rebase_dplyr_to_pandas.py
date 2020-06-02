import pandas as pd
import numpy as np
import re
from src.utils import _check_df_type, _get_list_columns, _get_str_columns


# One Table Verbs
def arrange(data, cols):
    """Arrange rows by column values

    Parameters
    ---------
    data: pandas or pysparkDataFrame
        A dataframe
    cols: str or list
        The columns we are sorting the dataframe on

    Returns
    -------
    Our sorted dataframe

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
    is_pandas = _check_df_type(data, "arrange")
    if isinstance(cols, str):
        columns = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, list):
        columns = _get_list_columns(data, cols, is_pandas)
    else:
        raise TypeError("Cannot determine method for determining column types")
    sorting_cols = []
    ascending_cols = []
    for c in columns:
        if re.search('desc()', c):
            sorting_cols.append(re.sub(r'desc|\(|\)|\s+', r'', c))
            ascending_cols.append(False)
        else:
            sorting_cols.append(c)
            ascending_cols.append(True)
    if is_pandas:
        return data.sort_values(sorting_cols, ascending=ascending_cols).reset_index().drop('index', axis=1)
    else:
        return data.orderBy(sorting_cols, ascending=ascending_cols)


def distinct(data, cols=None, keep_all=False):
    """Subset distinct/unique rows

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A dataframe
    cols: str or list, default is None
         Optional variables to use when determining uniqueness. If there are multiple rows for a given combination of inputs,
         only the first row will be preserved. If None, will use all variables.
    keep_all: bool, default is False
        If True, keep all variables in data. If a combination of cols is not distinct, this keeps the first row of values.

    Returns
    -------
    Our dataframe but with unique rows specified
    """
    is_pandas = _check_df_type(data, "distinct")
    if isinstance(cols, str):
        distinct_cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, list):
        distinct_cols = _get_list_columns(data, cols, is_pandas=is_pandas)
    else:
        if is_pandas:
            distinct_cols = data.columns
        else:
            distinct_cols = data.schema.names
    if is_pandas:
        if keep_all:
            dropped_data = data[distinct_cols].drop_duplicates(keep='first')
            dropped_data = pd.merge(dropped_data, data.drop(distinct_cols, axis=1), left_index=True, right_index=True, how='left')
            dropped_data.index = np.arange(len(dropped_data))
            return dropped_data
        else:
            return data[distinct_cols].drop_duplicates(keep='first', ignore_index=True)
    else:
        ...


def filter(data, cols):
    """Filters data based on arguments from args

    Parameters
    ----------
    data: pandas DataFrame
        The dataframe for which we filtering the data on
    *args: str
        The filter conditions we are applying on our dataframe

    Returns
    -------
    filtered_data: pandas DataFrame
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
    is_pandas = _check_df_type(data, "filter")
    if isinstance(cols, str):
        cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, list):
        cols = _get_list_columns(data, cols, is_pandas)
    query_result = []
    counter = 1
    args_length = len(cols)
    for c in cols:
        if "mean(" in c.casefold():
            mean_col = re.search(r'(?<=mean\()[a-zA-Z]+', c).group(0)
            if is_pandas:
                val = data[mean_col].mean()
            else:
                ...
            comparison = re.search(r'([<>]=?|==)', c).group(0)
            result = '{} {} {}'.format(mean_col, comparison, val)
        elif "median(" in c.casefold():
            median_col = re.search(r'(?<=median\()[a-zA-Z]+', c).group(0)
            if is_pandas:
                val = data[median_col].median()
            else:
                ...
            comparison = re.search(r'([<>]=?|==)', c).group(0)
            result = '{} {} {}'.format(median_col, comparison, val)
        elif "min(" in c.casefold():
            min_col = re.search(r'(?<=min\()[a-zA-Z]+', c).group(0)
            if is_pandas:
                val = data[min_col].min()
            else:
                ...
            comparison = re.search(r'([<>]=?|==)', c).group(0)
            result = '{} {} {}'.format(min_col, comparison, val)
        elif "max(" in c.casefold():
            max_col = re.search(r'(?<=max\()[a-zA-Z]+', c).group(0)
            if is_pandas:
                val = data[max_col].max()
            else:
                ...
            comparison = re.search(r'([<>]=?|==)', c).group(0)
            result = '{} {} {}'.format(max_col, comparison, val)
        elif "quantile(" in c.casefold():
            quantile_col = re.search(r'(?<=quantile\()[a-zA-Z]+', c).group(0)
            if re.search('probs=', c):
                quantile_percent = float(re.search(r'(?<=probs\=)\s*\d*\.\d+', c).group(0))
            else:
                quantile_percent = float(re.search(r'(?<=,)\s*\d*\.\d+', c).group(0))
            if quantile_percent > 1:
                raise Exception("Cannot have percentile greater than 1")
            comparison = re.search(r'([<>]=?|==)', c).group(0)
            if is_pandas:
                val = data[quantile_col].quantile(quantile_percent)
            else:
                ...
            result = '{} {} {}'.format(quantile_col, comparison, val)
        else:
            result = c
        if counter < args_length:
            if is_pandas:
                result = result + ' & '
            else:
                result = result + ' and '
        query_result.append(result)
        counter += 1
    if is_pandas:
        return data.query(''.join(query_result))
    else:
        return data.filter(''.join(query_result))


def pull(data, var, name=None):
    """Extracts a single column

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A dataframe
    var: int or str
        A variable specified as:
            * a literal variable name
            * a positive integer, giving the position counting from the left
            * a negative integer, giving the position counting from the right.
    name: int or str, default is None
        An optional parameter that specifies the column to be used as names for a named vector.
        Specified in a similar manner as var.

    Returns
    -------
    Our column we wish to extract
    """
    is_pandas = _check_df_type(data, "pull")
    if is_pandas:
        if isinstance(var, str):
            return_col = data[var]
        else:
            return_col = data.iloc[:, var]
        if isinstance(name, str):
            return_col.index = data[name]
        elif isinstance(name, int):
            return_col.index = data.iloc[:, name]
    else:
        if isinstance(var, str):
            return_col = data.select(var)
        else:
            return_col = data.select(data.columns[var])
    return return_col