import pandas as pd
from src.utils import _check_df_type


def everything(data):
    """Selects all variables

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A dataframe

    Returns
    -------
    All columns within our dataframe
    """
    is_pandas = _check_df_type(data, "everything")
    if is_pandas:
        return data.columns.tolist()
    else:
        return data.columns


def last_col(data, offset=0):
    """Selects the `nth` last column

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A dataframe
    offset: int, default=0
        Set it to `n` to select the nth var from the end

    Returns
    -------
    The `nth` last column from our data
    """
    is_pandas = _check_df_type(data, "last_col")
    return data.columns[-(offset + 1)]
