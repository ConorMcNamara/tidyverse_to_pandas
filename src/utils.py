from more_itertools import unique_everseen
import pandas as pd
import pyspark.sql as ps
import warnings


def _get_str_columns(data, str_arguments, cols=None, is_pandas=True):
    """Accounts for various tidyverse syntax that Hadley Wickham uses for selecting (or deselecting) columns

    Parameters
    ----------
    data: pandas or pyspark column
        Our data frame for determining applicable column names
    str_arguments: str
        The arguments for selecting columns. Currently supported are the name itself, "x", everything but the column
        listed, "-x", and every column in-between, "x:y".
    cols: optional, list of column names, default is None
        If given, we use these instead of calculating the names ourselves. Mostly used to simplify _get_list_columns
    is_pandas: bool, default is True
        Whether our DataFrame is in pandas or pyspark format

    Returns
    -------
    All applicable columns
    """
    if cols is None:
        if is_pandas:
            cols = data.columns.tolist()
        else:
            cols = data.columns
    if ":" in str_arguments:
        start_col, end_col = str_arguments.split(':')
        start_index, end_index = cols.index(start_col), cols.index(end_col) + 1
        cols = cols[start_index:end_index]
    elif "-" in str_arguments:
        col_to_remove = str_arguments[str_arguments.find("-") + 1:]
        cols.remove(col_to_remove)
    else:
        cols = [str_arguments]
    return cols


def _get_list_columns(data, list_cols, is_pandas=True):
    """Accounts for various tidyverse syntax that Hadley Wickham uses for selecting (or deselecting) columns

    Parameters
    ----------
    data: pandas or pyspark column
        Our data frame for determining applicable column names
    list_cls: list, tuple or numpy array
        The arguments for selecting columns. Currently supported are the name itself, "x", everything but the column
        listed, "-x", and every column in-between, "x:y".
    is_pandas: bool, default is True
        Whether our DataFrame is in pandas or pyspark format

    Returns
    -------
    All applicable columns
    """
    if is_pandas:
        cols = data.columns.tolist()
    else:
        cols = data.columns
    return_cols = []
    for column in list_cols:
        return_cols = return_cols + _get_str_columns(data, column, cols=cols)
    return list(unique_everseen(return_cols))


def _convert_numeric(data):
    # We go column by column and see if the column contains only numeric characters. If it does, then we
    # can safely use pd.to_numeric(), which also handles if it gets converted to integer or float.
    return data.apply(lambda x: pd.to_numeric(x) if x.str.isnumeric().all() else x, axis=0)


def _check_df_type(data, argument):
    if isinstance(data, pd.DataFrame):
        return True
    elif isinstance(data, ps.Column):
        return False
    else:
        raise Exception("Cannot perform {} on non-DataFrame".format(argument))


def _check_unique(data, how='unique'):
    # Check for repeated names
    if len(set(data.columns)) != len(data.columns):
        if how.casefold() == 'check_unique':
            raise AttributeError("Not all columns have unique names")
        elif how.casefold() == 'unique':
            cols = pd.Series(data.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in
                                                                 range(sum(cols == dup))]
            data.columns = cols
            return data
        else:
            warnings.warn("Not all columns have unique names")
            return data
    else:
        return data
