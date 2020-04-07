import pandas as pd
import pyspark.sql as ps
from pyspark.sql.functions import concat_ws
from src.utils import _get_list_columns, _convert_numeric, _get_str_columns, _check_df_type
import warnings
import numpy as np
from itertools import chain


# Pivoting


def pivot_longer(data, cols, names_to="name", names_prefix=None, names_sep=None, names_pattern=None,
                 names_ptypes=None, names_repair="check_unique", values_to="value", values_drop_na=True,
                 values_ptypes=None):
    """pivot_longer() "lengthens" data, increasing the number of rows and decreasing the number of columns.

    Parameters
    ----------
    data: pandas or pyspark dataframe
        A data frame to pivot.
    cols: str or list
        Columns to pivot into longer format.
    names_to: str or list, default is "name"
        A string specifying the name of the column to create from the data stored in the column names of data.
        Can be a list, creating multiple columns, if names_sep or names_pattern is provided.
    names_prefix: str, default is None
        A regular expression used to remove matching text from the start of each variable name.
    names_sep: list or str, default is None
        If names_to contains multiple values, these arguments control how the column name is broken up.
        names_sep takes the same specification as separate(), and can either be a numeric vector (specifying positions
        to break on), or a single string (specifying a regular expression to split on).
    names_pattern: list or str, default is None
        If names_to contains multiple values, these arguments control how the column name is broken up.
        names_pattern takes the same specification as extract(), a regular expression containing matching groups (()).

        If these arguments do not give you enough control, use pivot_longer_spec() to create a spec object and process
        manually as needed.
    names_ptypes
    names_repair
    values_to: str, default is None
        A string specifying the name of the column to create from the data stored in cell values.
        If names_to is None this value will be ignored, and the name of the value column will be derived from part of
        the existing column names.
    values_drop_na: bool, default is True
        If True, will drop rows that contain only NAs in the value_to column. This effectively converts explicit missing values
        to implicit missing values, and should generally be used only when missing values in data were created by its structure.
    values_ptypes

    Returns
    -------

    """
    is_pandas = _check_df_type(data, "pivot_longer")
    # Here, we check to see which columns we are going to "pivot on". That is, we're going to see which columns
    # are going to be used to create new, more compact variables; at the expense of more rows being created
    if isinstance(cols, str):
        cols = _get_str_columns(data, str_arguments=cols, is_pandas=is_pandas)
    elif isinstance(cols, (list, tuple)):
        cols = _get_list_columns(data, cols, is_pandas)
    else:
        raise TypeError("Cannot determine metod for determining column types")
    # Here, we are testing if the user wishes to create multiple columns. If so, then we need to see how the columns are
    # going to be created.
    if is_pandas:
        if isinstance(names_to, (list, tuple)):
            # Here, we are using name_sep, which indicates a separation using either a string, for example, "_" would indicate
            # that we split on underscores, or a numeric list, for example [1, 2, 3] would indicate that we split on indices
            # 1, 2 and 3.
            if names_sep is not None:
                if isinstance(names_sep, str):
                    splits = data[cols].apply(lambda x: x.str.split(pat=names_sep, expand=True))
                elif isinstance(names_sep, (tuple, list)):
                    splits = pd.DataFrame()
                    # str.slice() returns "" if there is no match, which will fail our covert condition, so we convert it to
                    # NaN instead to represent that there was no match
                    for index, name in enumerate(cols):
                        if index == 0:
                            splits[name] = data[cols].str.slice(stop=names_sep[index]).replace("", np.nan)
                        elif index == len(cols) - 1:
                            splits[name] = data[cols].str.slice(start=names_sep[index - 1]).replace("", np.nan)
                        else:
                            splits[name] = data[cols].str.slice(start=names_sep[index - 1], stop=names_sep[index]).replace("", np.nan)
                else:
                    raise TypeError("Cannot determine method of splitting DataFrame")
                ...
            elif names_pattern is not None:
                splits = data[cols].apply(lambda x: x.str.extract(names_pattern, expand=True))
                ...
            else:
                raise AttributeError("Cannot determine method of breaking up data into multiple columns")
        if values_to is None:
            ...



    def pivot_wider(data, id_cols, names_from=None, names_prefix="", names_sep="_", names_repair="check_unique",
                    values_from=None, values_fill=None, values_fn=None):


# Rectangling


# Nesting and Unnesting Data


def nest(data, cols=None):
    """Nesting creates a list-column of data frames

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        Our data frame
    cols: dictionary, default is None
        Name-variable pairs of the form {new_col: [col1, col2, col3]} that describe how you wish to nest
        existing columns into new columns.


    Returns
    -------
    Our dataframe, but with nested columns
    """
    is_pandas = _check_df_type(data, "nest")
    if not isinstance(cols, dict):
        raise TypeError("cols must be in dictionary format")
    keys = _get_list_columns(data, list(cols.keys()), is_pandas)
    vals = _get_list_columns(data, list(chain.from_iterable(list(cols.values()))), is_pandas)
    groupby_cols = data.columns.difference(keys + vals)
    for key in cols.keys():
        data[key] = data[cols[key]].values.tolist()


def unnest(data, cols, into, keep_empty=False, ptype=None, names_sep=None, names_repair="check_unique"):
    """

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        Our data frame
    cols: str or list
        Names of columns to unnest
    into: list or tuple
        Names of our unnested columns
    keep_empty: bool, default is False
        By default, you get one row of output for each element of the list your unnesting.
        This means that if there's a size-0 element (like np.nan or an empty data frame), that entire row will be dropped
        from the output. If you want to preserve all rows, use keep_empty=True to replace size-0 elements with a
        single row of missing values.
    ptype: pandas or pyspark DataFrame, default is None
        Optionally, supply a data frame prototype for the output cols, overriding the default that will be guessed
        from the combination of individual values.
    names_sep: str, default is None
        If None, the default, the names of new columns will come directly from our cols
        If a string, the names of the new columns will be formed by pasting together the outer column name with the
        inner names, separated by names_sep.
    names_repair: str, default is "check_unique"
        Used to check that output data frame has valid names. Must be one of the following options:
            "minimal": no name repair or checks, beyond basic existence,
            "unique": make sure names are unique and not empty,
            "check_unique": (the default), no name repair, but check they are unique,
            "universal": make the names unique and syntactic

    Returns
    -------

    """
    is_pandas = _check_df_type(data, "unnest")
    if is_pandas:
        # Here, we replace all empty lists as NAs
        data = data.mask(data.applymap(type).eq(list) & ~data.astype(bool))
        # Here, we replace the NAs as a list of NAs, with our list length as determined by the maximum of each row
        if keep_empty is True:
            for col in cols:
                max_val = data[col].map(len).max()
                data[col] = data[col].fillna([np.nan] * max_val)
        # If keep_empty = False, we instead drop the NAs from our analysis
        else:
            data = data.dropna(how='any', axis=0, subset=cols)
        # Single new variable created
        if isinstance(cols, str):
            if names_sep is not None:
                into = ['{}{}{}'.format(cols, names_sep, column) for column in into]
            if len(data) < 500000:
                # This is much faster, but has problems scaling to large data frames
                temp_df = pd.DataFrame(data[cols].values.tolist(), columns=into)
            else:
                # This is much slower, but scales much better to large data frames
                temp_df = data[cols].apply(pd.Series)
                temp_df.columns = into
        # Multiple new variables created
        elif isinstance(cols, (tuple, list)):
            if names_sep is not None:
                into = ['{}{}{}'.format(cols, names_sep, column) for row in into for column in row]
            temp_df = pd.DataFrame()
            if len(data) < 500000:
                for index, col in enumerate(cols):
                    # Again, this is much faster, but scales poorly to large data frames
                    cols_expand = pd.DataFrame(data[col].values.tolist(), columns=into[index])
                    temp_df = pd.concat([temp_df, cols_expand], axis=0)
            else:
                for index, col in enumerate(cols):
                    # Similarly, this is much slower but scales much better to large data frames
                    cols_expand = data[col].apply(pd.Series)
                    cols_expand.columns = into[index]
                    temp_df = pd.concat([temp_df, cols_expand], axis=0)
        else:
            raise TypeError("Cannot determine method of unnesting columns")




# Splitting and Combining Character Columns


def separate(data, col, into, sep="_", remove=True, convert=False, extra="warn", fill="warn"):
    """Given either regular expression or a vector of character positions, separate() turns a single character column
    into multiple columns.

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame
    col: str
        Column name or position.
    into: list
        Names of new variables to create as character vector. Use "NA" to omit the variable in the output, e.g.
        ['NA', 'B]
    sep: str or int
        Separator between columns.
        If character, is interpreted as a regular expression. The default value is a regular expression that matches any
        sequence of non-alphanumeric values.
        If list or tuple, interpreted as positions to split at. Positive values start at 0 at the far-left of the string;
        negative value start at -1 at the far-right of the string. The length of sep should be one less than into.
    remove: bool, default is True
        If True, remove input column from output data frame.
    convert: bool, default is False
        If True, will run pd.to_numeric() if applicable on new columns.
        This is useful if the component columns are integer, numeric or logical.
    extra: str, options are {"warn", "drop", "merge"}, default is "warn"
        "warn" (the default): emit a warning and drop extra values.
        "drop": drop any extra values without a warning.
        "merge": only splits at most len(into) - 1 times
    fill: str, options are {"warn", "right", "left"}, default is "warn"
        "warn" (the default): emit a warning and fill from the right
        "right": fill with missing values on the right
        "left": fill with missing values on the left

    Returns
    -------
    Our dataframe, but with separated character columns
    """
    is_pandas = _check_df_type(data, "separate")
    extra, fill = extra.casefold(), fill.casefold()
    if is_pandas:
        if isinstance(sep, str):
            # 0 for str.split() means we run as many splits as possible, whereas num_times indicates the maximum
            # number of splits we are allowing
            num_times = len(into) - 1 if extra == 'merge' else 0
            splits = data[col].str.split(pat=sep, n=num_times, expand=False)
            # Here, one issue is that str.split() returns NaN for splits with no result. However, NaN returns an error
            # when trying to calculate its length. So we are converting all instances of NaN to an empty list
            splits = splits.apply(lambda d: d if isinstance(d, list) else [])
            # For str.split(), if we specify that expand=False, our results are returned as a Series of lists instead
            # of a pandas DataFrame. Thus, we check if the length of each list in our Series is equal to the number
            # of column names we want to create. If it's less, we add NAs to the list until the lengths are equivalent,
            # with the fill() argument specifying where the NAs will be placed (either right or left of our non-NA values)
            if fill == 'left':
                splits = splits.apply(lambda x: x if len(x) == len(into) else [np.nan] * (len(into) - len(x)) + x)
            else:
                if fill == 'warn':
                    warnings.warn('Filling values from right to left')
                for i in range(len(splits)):
                    splits = splits.apply(lambda x: x if len(x) == len(into) else x + [np.nan] * (len(into) - len(x)))
            # Turn our Series of lists into a DataFrame
            splits = pd.DataFrame(item for item in splits)
            # This accounts for the opposite problem, which is that the length of the lists in our Series is greater
            # than the number of column names we want to create. We name the excess columns as NA, and then drop them
            diff = splits.shape[1] - len(into)
            if extra == 'warn':
                if diff > 0:
                    warnings.warn("Expected {} pieces. Additional piece discarded in {} rows {}".format(len(into), diff,
                                                                                                        [i for i in
                                                                                                         range(
                                                                                                             splits.shape[
                                                                                                                 0],
                                                                                                             splits.shape[
                                                                                                                 0] + diff)]))
            splits.columns = into + ['NA'] * diff
            # We do this instead of checking if diff > 0 because the user could've specified certain columns to be
            # removed within into() by calling them "NA"
            if 'NA' in splits.columns:
                splits = splits.drop(["NA"], axis=1)
        elif isinstance(sep, (list, tuple)):
            # Unlike using character separators, numeric ones do not allow for filling or other error handling, so
            # we need the number of column names to exactly match how many separations we are going to create
            if len(sep) != len(into) - 1:
                raise AttributeError(
                    "{} column names were expected but {} column names were received".format(len(into) - 1, len(sep)))
            splits = pd.DataFrame()
            # str.slice() returns "" if there is no match, which will fail our covert condition, so we convert it to
            # NaN instead to represent that there was no match
            for index, name in enumerate(into):
                if index == 0:
                    splits[name] = data[col].str.slice(stop=sep[index]).replace("", np.nan)
                elif index == len(into) - 1:
                    splits[name] = data[col].str.slice(start=sep[index - 1]).replace("", np.nan)
                else:
                    splits[name] = data[col].str.slice(start=sep[index - 1], stop=sep[index]).replace("", np.nan)
        else:
            raise ValueError("Cannot determine method of splitting columns")
        if convert:
            splits = _convert_numeric(splits)
        # We add the splitted values to our dataframe
        data = pd.concat([data, splits], axis=1)
        if remove:
            data = data.drop([col], axis=1)
    else:
        # pyspark has no native ability to extract column names, so we'll need to do it ourselves
        if remove:
            data = data.drop(col)
    return data


def extract(data, col, into, regex="([a-zA-Z0-9]+)", remove=True, convert=False):
    """Given a regular expression with capturing groups, extract() turns each group into a new column.
    If the groups don't match, or the input is NA, the output will be NA.

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame.
    col: str
        Column name or position
    into: list
        Names of new variables to create as character vector. Use "NA" to omit the variable in the output, e.g.
        ["NA", "B"]
    regex: str, default is all alpha-numeric characters
        A regular expression used to extract the desired values.
        There should be one capture group (defined by ()) for each element of into.
    remove: bool, default is True
        If True, remove input column from output DataFrame
    convert: bool, default is False
        If True, will run pd.to_numeric() if applicable on new columns.
        This is useful if the component columns are integer, numeric or logical.

    Returns
    -------
    Our dataframe, but with extracted character columns
    """
    is_pandas = _check_df_type(data, "extract")
    if is_pandas:
        # Here, we are setting expand=True so that pandas returns a DataFrame instead of a Series of lists
        splits = data[col].str.extract(regex, expand=True)
        # If our regex extracts more columns column names provided, then we are keeping the first n columns,
        # where n is the number of column names provided
        if len(into) < splits.shape[1]:
            splits = splits.iloc[:, 0:len(into)]
        splits.columns = into
        # If user specifies they want certain columns to be dropped
        if "NA" in splits.columns:
            splits.drop(['NA'], axis=1)
        if convert:
            splits = _convert_numeric(splits)
        data = pd.concat([data, splits], axis=1)
        if remove:
            data = data.drop([col], axis=1)
    else:
        # use Pyspark regular expressions
        if remove:
            data = data.drop(col)
    return data


def unite(data, col, input_cols=None, sep='_', remove=True, na_rm=False):
    """Convenience function to paste together multiple columns into one.

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame.
    col: str
        The name of our new column
    input_cols: list, default is None
        A selection of columns. If None, all variables are selected. You can supply bare variable names,
        select all variables between x and z with x:z, or exclude y with -y
    sep: str, default is '_"
        Separator to use between values.
    remove: bool, default is True
        If True, remove input columns from output data frame.
    na_rm: bool, default is True
        If True, missing values will be remove prior to uniting each value.

    Returns
    -------
    Our dataframe, but with united columns.
    """
    is_pandas = _check_df_type(data, "unite")
    if input_cols is None:
        if is_pandas:
            input_cols = data.columns.tolist()
        else:
            input_cols = data.columns
    if not isinstance(input_cols, (list, tuple)):
        raise ValueError("Cannot determine which columns to unite")
    input_cols = _get_list_columns(data, list_cols=input_cols, is_pandas=is_pandas)
    if is_pandas:
        # If we were to simply run .str.cat(), it would return NaN if any of the columns it tried to concatenate had
        # had a NaN. So, we run fillna() to convert NAs to strings, then later run replace to convert that NA back to
        # NaN.
        data = data.fillna('NA')
        data[col] = data[input_cols[0]].str.cat(data[input_cols[1:]], sep=sep)
        if na_rm:
            # pandas does not let us selectively remove values/columns when it comes to using str.concat(). So I thought
            # of two ways of solving it
            # 1) replace NAs with empty strings, and then perform concatenate on that. However, trying to get the
            # replacements means using regex to separate "a_" versus "a_b", which is less than ideal.
            # 2) replace the concatenated values after the fact. Since we don't need regex, seems like a much more stable
            # solution
            data[col] = data[col].str.replace('NA_', "").str.replace("_NA", "").str.replace("NA", "")
        if remove:
            data = data.drop(input_cols, axis=1)
        data = data.replace("NA", np.nan)
    else:
        if na_rm:
            data = data.dropna(how='any', subset=input_cols)
        data = data.withColumn(col, concat_ws(sep, *input_cols))
        if remove:
            data = data.drop(col)
    return data


# Dealing with missing data


def drop_na(data, cols=None):
    """Drop rows containing missing values

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame.
    cols: list of str or str, default is None
        A selection of columns. If None, all variables are selected.
        If in string format, expected to either be a singular column, such as "x" or "-y" or in "x:z" format if
        selecting multiple columns. Note that for "x:y", we are assuming every character prior to : is the column name,
        so "x :z" will expect the column to be named "x ".
        If in list format, expected to be a list of strings containing column names, such as ["x", "y"] or ["x", "-y"]

    Returns
    -------
    Our dataframe, but with NAs removed
    """
    is_pandas = _check_df_type(data, "drop_na")
    if cols is None:  # Drop any row that contains a NA
        if is_pandas:
            data = data.dropna(axis=0, how='any')
        if not is_pandas:
            data = data.dropna(how='any')
    else:  # Drop any row that contains an NA within certain columns
        if isinstance(cols, str):
            cols_to_consider = _get_str_columns(data, cols, is_pandas=is_pandas)
        else:
            cols_to_consider = _get_list_columns(data, cols, is_pandas)
        if is_pandas:
            data = data.dropna(axis=0, how='any', subset=cols_to_consider)
        else:
            data = data.dropna(how='any', subset=cols_to_consider)
    return data


def replace_na(data, replace):
    """Replace missing values.

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame
    replace: str or int or dict
        If str or int, then replacing every instance of a NA with given value
        If dict, then replacing each given column (identified as the key)'s NA with the value pair.

    Returns
    -------
    Our dataframe, but with NAs filled. Note that integer columns are going to be returned as float columns
    """
    if isinstance(data, pd.DataFrame):
        data = data.fillna(replace, axis=0)
    elif isinstance(data, ps.DataFrame):
        data = data.fillna(replace)
    else:
        raise Exception("Cannot fill NAs on a non-DataFrame")
    return data


def fill(data, cols=None, direction='down'):
    """Fills missing values in selected columns using the next or previous entry.
    This is useful in the common output format where values are not repeated, and are only recorded when they change.

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame
    cols: str or list, default is None
        A selection of columns. If None, nothing happens.
        If in string format, expected to either be a singular column, such as "x" or "-y" or in "x:z" format if
        selecting multiple columns. Note that for "x:y", we are assuming every character prior to : is the column name,
        so "x :z" will expect the column to be named "x ".
        If in list format, expected to be a list of strings containing column names, such as ["x", "y"] or ["x", "-y"]
    direction: str, options are {"down", "up", "downup", "updown"}, default is 'down'
        Direction in which to fill missing values. Currently either "down" (the default), "up",
        "downup" (i.e. first down and then up) or "updown" (first up and then down).

    Returns
    -------
    Our dataframe, but with missing values filled. Note that integers get converted to floats in pandas
    """
    direction = direction.casefold()
    if direction not in ['down', 'up', 'downup', 'updown']:
        raise ValueError("Cannot identify method of filling missing data")
    if cols is None:
        return data
    is_pandas = _check_df_type(data, "fill")
    if isinstance(cols, str):
        cols_to_consider = _get_str_columns(data, cols, is_pandas=is_pandas)
    else:
        cols_to_consider = _get_list_columns(data, cols, is_pandas=is_pandas)
    if is_pandas:
        if direction == "down":
            data[cols_to_consider] = data[cols_to_consider].fillna(method='ffill')
        elif direction == "up":
            data[cols_to_consider] = data[cols_to_consider].fillna(method='bfill')
        elif direction == 'downup':
            data[cols_to_consider] = data[cols_to_consider].fillna(method='ffill').fillna(method='bfill')
        else:
            data[cols_to_consider] = data[cols_to_consider].fillna(method='bfill').fillna(method='ffill')
    else:
        # Pyspark has no native support for forwards or backwards filling, so need to create my own function to do it
        ...
    return data


def complete():
    ...