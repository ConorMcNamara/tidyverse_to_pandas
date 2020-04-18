import pandas as pd
import pyspark.sql as ps
from pyspark.sql.functions import concat_ws
from src.utils import _get_list_columns, _convert_numeric, _get_str_columns, _check_df_type, _check_unique
import warnings
import numpy as np
import re


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
    names_ptypes: list, default is None
        A list of column name-prototype pairs. A prototype (or ptype for short) is a zero-length vector
        (like integer() or numeric()) that defines the type, class, and attributes of a vector.

        If not specified, the type of the columns generated from names_to will be character.
    names_repair: str, default is "check_unique"
        What happen if the output has invalid column names?
        The default, "check_unique" is to error if the columns are duplicated.
        Use "minimal" to allow duplicates in the output, or "unique" to de-duplicated by adding numeric suffixes.
    values_to: str, default is None
        A string specifying the name of the column to create from the data stored in cell values.
        If names_to is None this value will be ignored, and the name of the value column will be "value".
    values_drop_na: bool, default is True
        If True, will drop rows that contain only NAs in the value_to column. This effectively converts explicit missing values
        to implicit missing values, and should generally be used only when missing values in data were created by its structure.
    values_ptypes: list, default is None
        The type of data our values_to column will be. Examples include int, str, np.int16 or bool.

        If not specified, the type of the variables generated from values_to will be the common type of the input columns
        used to generate them.

    Returns
    -------
    Our melted/longer pivoted data frame
    """
    is_pandas = _check_df_type(data, "pivot_longer")
    values_to = values_to if values_to is not None else "value"
    names_to = names_to if names_to is not None else "variable"
    if is_pandas:
        # Single column to pivot/melt on
        if isinstance(cols, str):
            cols = _get_str_columns(data, str_arguments=cols, is_pandas=is_pandas)
            # We need to see which columns we are going to melt/pivot longer on, so we take the "difference" of our
            # data frame's columns from the columns that we know we won't be pivoting on.
            id_vars = data.columns.difference(cols)
        # Multiple columns to pivot/melt on
        elif isinstance(cols, (list, tuple)):
            cols = _get_list_columns(data, cols, is_pandas)
            id_vars = data.columns.difference(cols)
        else:
            raise TypeError("Cannot determine method for determining column types")
        if isinstance(names_to, str):
            # This is by far the easiest, where everything is specified for us. The names are strings, the values are
            # strings, and everything is easy to calculate
            melted_data = pd.melt(data, id_vars=id_vars, value_vars=cols, var_name=names_to, value_name=values_to)
        else:
            # Here, we need to figure out what to name our variables as, as well as how to separate them.
            melted_data = pd.melt(data, id_vars=id_vars, value_vars=cols, value_name=values_to)
            # Uses similar argumentation to separate
            if names_sep is not None:
                melted_data = separate(melted_data, "variable", names_to, names_sep, remove=True, convert=False,
                                       extra="drop", fill="right")
            # Uses similar arguments to extract
            elif names_pattern is not None:
                melted_data = extract(melted_data, 'variable', names_to, names_pattern, remove=True, convert=False)
            else:
                raise ValueError("Cannot determine how to separate into multiple columns")
        # Drop NA values
        if values_drop_na:
            melted_data = melted_data.dropna(how='any', axis=0)
        # Remove prefixes from our names data
        if names_prefix is not None:
            melted_data[names_to] = melted_data[names_to].str.replace(names_prefix, "")
        # Check for repeated names
        melted_data = _check_unique(melted_data, how=names_repair)
        # Convert pytypes
        if names_ptypes is not None:
            conversion_type = {}
            for index, column in enumerate(names_to):
                conversion_type[column] = names_ptypes[index]
            melted_data = melted_data.astype(conversion_type)
        if values_ptypes is not None:
            melted_data[values_to] = melted_data[values_to].astype(values_ptypes)
    else:
        ...
    return melted_data


def pivot_wider(data, id_cols=None, names_from="name", names_prefix="", names_sep="_", names_repair="check_unique",
                values_from="value", values_fill=None, values_fn=None):
    """pivot_wider() "widens" data, increasing the number of columns and decreasing the number of rows.
    The inverse transformation is pivot_longer().

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame to pivot.
    id_cols: str or list, default is None
        A set of columns that uniquely identifies each observation.
        Defaults to all columns in data except for the columns specified in names_from and values_from.
        Typically used when you have additional variables that is directly related.
    names_from: str, default is "name"
        Describes which column to get the name of the output column.
    names_prefix: str, default is ""
        String added to the start of every variable name. This is particularly useful if names_from is a numeric vector
        and you want to create syntactic variable names.
    names_sep: str, default is "_"
        If names_from or values_from contains multiple variables, this will be used to join their values together
        into a single string to use as a column name.
    names_repair: str, default is "check_unique"
        What happen if the output has invalid column names? The default, "check_unique" is to error if the columns are duplicated.
        Use "minimal" to allow duplicates in the output, or "unique" to de-duplicated by adding numeric suffixes.
    values_from: str or list, default is "name"
        Describes which column (or columns) to get the cell values from.
    values_fill: dict, default is None
        Optionally, a dictionary specifying what each value should be filled in with when missing.
    values_fn: list, default is None
        Optionally, a named list providing a function that will be applied to the value in each cell in the output.
        You will typically use this when the combination of id_cols and value column does not uniquely identify an observation.

    Returns
    -------
    Our pivoted dataframe
    """
    is_pandas = _check_df_type(data, "pivot_wider")
    if isinstance(names_from, str):
        names_from = _get_str_columns(data, names_from, is_pandas=is_pandas)
    elif isinstance(names_from, (tuple, list)):
        names_from = _get_list_columns(data, names_from, is_pandas)
    else:
        raise TypeError("Cannot determine column names to be pivotted off of")
    if isinstance(values_from, str):
        values_from = _get_str_columns(data, values_from, is_pandas=is_pandas)
    elif isinstance(values_from, (tuple, list)):
        values_from = _get_list_columns(data, values_from, is_pandas=is_pandas)
    else:
        raise TypeError("Cannot determine value names to pivot off of")
    if is_pandas:
        if id_cols is None:
            id_cols = data.columns.difference(list(names_from) + list(values_from))
        if names_sep is not None and len(names_from) > 1:
            new_col = '{}'.format(names_sep).join([name for name in names_from])
            data = unite(data, new_col, input_cols=names_from, sep=names_sep, remove=True, na_rm=False)
            names_from = new_col
        if names_sep is not None and len(values_from) > 1:
            new_col = '{}'.format(names_sep).join([name for name in values_from])
            data = unite(data, new_col, input_cols=values_from, sep=names_sep, remove=True, na_rm=False)
            values_from = new_col
        if len(id_cols) > 1:
            pivoted_data = pd.pivot_table(data, index=id_cols, columns=names_from, values=values_from).reset_index()
        else:
            pivoted_data = pd.pivot(data, index=id_cols[0], columns=names_from[0], values=values_from)
        # Handle adding prefixes to necessary columns
        value_cols = [names_prefix + col for col in pivoted_data.columns.difference(list(id_cols))]
        pivoted_data[pivoted_data.columns.difference(list(id_cols))].columns = value_cols
        # Check for repeated names
        pivoted_data = _check_unique(pivoted_data, how=names_repair)

    else:
        ...
    return pivoted_data


# Rectangling


# Nesting and Unnesting Data


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
        ...
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
        ...
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
        ...
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


def complete(data, cols=None, fill=None):
    """Turns implicit missing values into explicit missing values.

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A data frame
    cols: list, default is None
        Specification of columns to expand
    fill: dictionary, default is None
        A dictionary that for each variable supplies a single value to use instead of NA for missing combinations.

    Returns
    -------

    """
    is_pandas = _check_df_type(data, "complete")
    # For nesting, we need to compute the possible combinations within each group, as pandas default is to find
    # every possible combination for every column, which isn't always the behavior we want (hence, nesting).
    if "nesting(" in cols:
        index_cols = [col for col in cols if 'nesting(' not in col]
        index_cols = _get_list_columns(data, index_cols, is_pandas)
        nesting_col = data.columns[data.columns.str.contains('nesting(')][0]
        nested_cols = re.search(r'\((.*?)\)', nesting_col).group(1).split(',').replace(" ", "")
    else:
        index_cols = cols
        nested_cols = None
    if is_pandas:
        data = data.set_index(index_cols)
        mux = pd.MultiIndex.from_product([data.index.levels[i] for i in range(len(index_cols))], names=index_cols)
        data = data.reindex(mux).reset_index()
        if fill is not None:
            data = data.fillna(fill)
    else:
        ...
        if fill is not None:
            data = data.fillna(fill)
    return data
