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
    data: pandas or pyspark column
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
        elif isinstance(cols, (list, tuple, (np.ndarray, np.generic))):
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
        pass
    return melted_data


def pivot_wider(data, id_cols=None, names_from="name", names_prefix="", names_sep="_", names_repair="check_unique",
                values_from="value", values_fill=None, values_fn=None):
    """pivot_wider() "widens" data, increasing the number of columns and decreasing the number of rows.
    The inverse transformation is pivot_longer().

    Parameters
    ----------
    data: pandas or pyspark column
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
    values_fn: dict, default is None
        Optionally, a dictionary providing a function that will be applied to the value in each cell in the output.
        You will typically use this when the combination of id_cols and value column does not uniquely identify an observation.

    Returns
    -------
    Our pivoted dataframe
    """
    is_pandas = _check_df_type(data, "pivot_wider")
    if isinstance(names_from, str):
        names_from = _get_str_columns(data, names_from, is_pandas=is_pandas)
    elif isinstance(names_from, (tuple, list, (np.ndarray, np.generic))):
        names_from = _get_list_columns(data, names_from, is_pandas)
    else:
        raise TypeError("Cannot determine column names to be pivotted off of")
    if isinstance(values_from, str):
        values_from = _get_str_columns(data, values_from, is_pandas=is_pandas)
    elif isinstance(values_from, (tuple, list, (np.ndarray, np.generic))):
        values_from = _get_list_columns(data, values_from, is_pandas=is_pandas)
    else:
        raise TypeError("Cannot determine value names to pivot off of")
    if is_pandas:
        if id_cols is None:
            id_cols = data.columns.difference(list(names_from) + list(values_from)).tolist()
        elif isinstance(id_cols, str):
            id_cols = _get_str_columns(data, id_cols, is_pandas=is_pandas)
        else:
            id_cols = _get_list_columns(data, id_cols, is_pandas)
        # Unites all naming columns into one larger column
        if names_sep is not None and len(names_from) > 1:
            new_col = '{}'.format(names_sep).join([name for name in names_from])
            data = unite(data, new_col, input_cols=names_from, sep=names_sep, remove=True, na_rm=False)
            names_from = [new_col]
        # Unites all value columns into one larger column
        if names_sep is not None and len(values_from) > 1:
            new_col = '{}'.format(names_sep).join([name for name in values_from])
            data = unite(data, new_col, input_cols=values_from, sep=names_sep, remove=True, na_rm=False)
            values_from = [new_col]
        if len(id_cols) > 1:
            pivoted_data = pd.pivot_table(data, index=id_cols, columns=names_from, values=values_from)
        else:
            # We take the 0th element from id_cols and names_from because pd.pivot() takes in strings as arguments, not
            # lists, so we need to get the string element within each list
            pivoted_data = pd.pivot(data, index=id_cols[0], columns=names_from[0], values=values_from)
        # Filling missing values
        if values_fill is not None:
            for column, fill_value in values_fill.items():
                pivoted_data.loc[:, pivoted_data.columns.get_level_values(0) == column] = pivoted_data.loc[:, pivoted_data.columns.get_level_values(0) == column].fillna(fill_value)
        # Flattening our columns names
        pivoted_data.columns = pivoted_data.columns.get_level_values(1)
        # Adding our id_cols back as variables instead of indexes, and resetting the index name to None
        pivoted_data = pivoted_data.reset_index()
        pivoted_data.columns.name = None
        # Re-order our dataframe so that the values are in the order first seen in our names_from column
        pivoted_data = pivoted_data[id_cols + data[names_from[0]].unique().tolist()]
        # Adding prefixes to names
        value_cols = [names_prefix + col for col in pivoted_data.columns.difference(id_cols)]
        pivoted_data[pivoted_data.columns.difference(id_cols)].columns = value_cols
        # Check for repeated names
        pivoted_data = _check_unique(pivoted_data, how=names_repair)
    else:
        pass
    return pivoted_data

# Rectangling Data


def unnest_longer(data, col, values_to=None, indices_to=None, indices_include=False, names_repair="check_unique", simplify=False, ptype=None):
    """Turns each element of a list-column into a row

    Parameters
    ----------
    data: pandas or pyspark column
        A data frame
    col: str or list
        Column to extract components from
    values_to: str or list, default is None
        Name of column to store vector values. If None, use col
    indices_to: str, default is None
        A string giving the name of column which will contain the inner names or position (if not named) of the values.
        If None, use col with _id suffix
    indices_include: bool, default is False
        Add an index column? Defaults to False when col has inner names.
    names_repair: str, default is check_unique
        "minimal": no name repair or checks, beyond basic existence,
        "unique": make sure names are unique and not empty,
        "check_unique": (the default), no name repair, but check they are unique
    simplify: bool, default is False
        If True, will attempt to simplify lists of length-1 vectors to an atomic vector
    ptype: dict, default is None
        Optionally, supply a dictionary for the output cols, overriding the default that will be guessed from
        the combination of individual values.

    Returns
    -------
    Our dataset, but with the nested columns expanded row-wise.
    """
    is_pandas = _check_df_type(data, "unnest_longer")
    if is_pandas:
        # We have multiple columns to explode. Note that this assumes that these columns are all columns of lists
        # and that each list contains the same number of elements
        if isinstance(col, (tuple, list, (np.ndarray, np.generic))):
            unnest_data = pd.concat([pd.DataFrame({x: np.concatenate(data[x].values)}) for x in col], axis=1)
            if indices_include:
                idx = data.index.repeat(data[col[0]].str.len())
                unnest_data.index = idx
                unnest_data = unnest_data.reset_index()
            unnest_data = unnest_data.join(data.drop(col, axis=1), how='left')
            if simplify:
                unnest_data[col] = unnest_data[col].apply(pd.to_numeric, errors='coerce')
        elif isinstance(col, str):
            # Our column is in list format
            if (data[col].apply(type) == list).any():
                unnest_data = data.explode(col)
            # Our column is in json/dict format
            else:
                unnest_data = pd.json_normalize(data[col])
                pivot_cols = list(unnest_data.columns)
                # Data is in wide format, so we need to convert it to long format
                unnest_data = unnest_data.join(data.drop(list(col), axis=1), how='left')
                indices_to = indices_to if indices_to is not None else '{}_id'.format(col)
                unnest_data = pivot_longer(unnest_data, pivot_cols, names_to=indices_to, values_to=col)
                # Our pivot longer returns the ids first, then the values. However, unnest_longer returns values first,
                # followed by id, so we need to reverse the column order to follow tidyr convention for this function
                unnest_data = unnest_data[list(data.columns.difference(list(col))) + list(col) + [indices_to]]
            if indices_include:
                unnest_data = unnest_data.reset_index()
            if simplify:
                unnest_data[col] = pd.to_numeric(unnest_data[col], errors='coerce')
        else:
            raise TypeError("Cannot determine columns to unnest dataframe")
        if values_to is not None:
            unnest_data[unnest_data.columns.difference(col)].columns = values_to
        if ptype is not None:
            unnest_data = unnest_data.astype(ptype)
        unnest_data = _check_unique(unnest_data, how=names_repair)
        # tidyr returns the data sorted by the columns not used to unnest, so we want to match that
        unnest_data = unnest_data.sort_values(list(data.columns.difference(list(col))))
        # Occasionally our merges will cause the indexes to not be consistent (for example, it may go 0, 1, 2, 3, 5);
        # so we set it back to consistent formatting
        unnest_data.index = np.arange(len(unnest_data))

    else:
        ...
    return unnest_data


def unnest_wider(data, col, names_sep=None, simplify=False, names_repair='check_unique', ptype=None):
    """Turns each element of a list-column into a column

    Parameters
    ----------
    data: pandas or pyspark column
        A data frame.
    col: str or list
        List-column to extract components from.
    names_sep: str, default is None
        If None, the default, the names of new columns will come directly from the inner json.
        If a string, the names of the new columns will be formed by pasting together the outer column name with the
         inner names, separated by names_sep.
    simplify: bool, default is False
        If True, will attempt to simplify lists of length-1 vectors to an atomic vector
    names_repair: str, default is check_unique
        "minimal": no name repair or checks, beyond basic existence,
        "unique": make sure names are unique and not empty,
        "check_unique": (the default), no name repair, but check they are unique
    ptype: dict, default is None
        Optionally, supply a dictionary for the output cols, overriding the default that will be guessed from
        the combination of individual values.

    Returns
    -------
    Our dataset, but with our nested columns expanded column-wise
    """
    is_pandas = _check_df_type(data, "unnest_wider")
    names_sep = names_sep if names_sep is not None else ''
    if is_pandas:
        if isinstance(col, (list, tuple, (np.ndarray, np.generic))):
            # Here, we are checking if all the columns contain the same length of lists. If they do, we can speed things
            # up by using data[col].tolist(). If they don't, we use data[col].apply(pd.Series).
            is_equal = True
            for c in col:
                if (data[c].str.len()[0] == data[c].str.len()[1:]).all():
                    continue
                else:
                    is_equal = False
                    break
            if is_equal:
                unnested_data = pd.concat([pd.DataFrame(data[x].tolist(), index=data.index).add_prefix('{}{}'.format(x, names_sep)) for x in col], axis=1)
            else:
                unnested_data = pd.concat([pd.DataFrame(data[x].apply(pd.Series), index=data.index).add_prefix('{}{}'.format(x, names_sep)) for x in col], axis=1)
        elif isinstance(col, str):
            # data is in list format
            if (data[col].apply(type) == list).any():
                # Similarly, we are checking if all the lists within the column are the same length. If they are, we can
                # speed things up using data[col].tolist(). Else, we use data[col].apply(pd.Series).
                if (data[col].str.len()[0] == data[col].str.len()[1:]).all():
                    unnested_data = pd.DataFrame(data[col].tolist()).add_prefix('{}{}'.format(col, names_sep))
                else:
                    unnested_data = data[col].apply(pd.Series).add_prefix('{}{}'.format(col, names_sep))
            # data is in dictionary format
            else:
                unnested_data = pd.json_normalize(data[col])
        else:
            raise TypeError("Cannot determine which columns to unnest on")
        if simplify:
            unnested_data = unnested_data.apply(pd.to_numeric, errors='coerce')
        # Merge our regular data with our unnested data, but with the regular data first
        unnested_data = data.drop(col, axis=1).join(unnested_data, how='left')
        # Check that the names for the columns follow our conventions
        unnested_data = _check_unique(unnested_data, how=names_repair)
        # Change pytypes if necessary
        if ptype is not None:
            unnested_data = unnested_data.astype(ptype)
    else:
        ...
    return unnested_data


# Nesting and Unnesting Data

def nest(data, cols):
    """Nesting transforms multiple rows and columns into nested dictionaries

    Parameters
    ----------
    data: pandas or pyspark column
        A data frame
    cols: str or list or dict
        Names of columns to nest. If dict, also specifies what the new column names will be.

    Returns
    -------
    Our dataframe, but with nested columns
    """
    is_pandas = _check_df_type(data, "nest")
    if isinstance(cols, str):
        cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, (list, tuple, (np.ndarray, np.generic))):
        cols = _get_list_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, dict):
        nested_cols = []
        for key, value in cols.items():
            cols[key] = _get_list_columns(data, value, is_pandas=is_pandas)
            nested_cols += cols[key]
    else:
        raise TypeError("Cannot determine columns to nest")
    if is_pandas:
        # Cols is in string or list format
        if isinstance(cols, (str, list, tuple, (np.ndarray, np.generic))):
            non_nested_cols = data.columns.difference(cols).tolist()
            # If every group in non_nested_cols has the same number of observations, then we can run our groupby and
            # lambda functions
            if (data.groupby(non_nested_cols).agg('count').iloc[0, 0] == data.groupby(non_nested_cols).agg('count').iloc[1:, 0]).all():
                nested_df = data.groupby(non_nested_cols).apply(lambda x: x.to_dict()).reset_index()
                nested_df = nested_df.rename({0: 'data'}, axis=1)
            # If our groups have different number of observations, then we need to iteratively add their dictionary
            # values as groupby and lambda will only select one value instead of all of them
            else:
                # Ensures that we don't have any unnecesary copies
                nested_df = data[non_nested_cols].drop_duplicates()
                nested_df['data'] = 0
                for index, row in nested_df.iterrows():
                    group = row[non_nested_cols].values
                    filtered_data = data[(data[non_nested_cols] == group).values]
                    nested_df.loc[index, 'data'] = [filtered_data[cols].to_dict()]
        # Cols in dictionary format
        else:
            non_nested_cols = data.columns.difference(nested_cols).tolist()
            nested_df = data[non_nested_cols].drop_duplicates()
            nested_df.index = np.arange(len(nested_df))
            # Similar to above, we are checking if every group in non_nested_cols has the same number of observations to
            # run our groupby and lambda expression successfully
            if (data.groupby(non_nested_cols).agg('count').iloc[0, 0] == data.groupby(non_nested_cols).agg('count').iloc[1:, 0]).all():
                for col_name, col_select in cols.items():
                    nested_col = data.groupby(non_nested_cols)[col_select].apply(lambda x: x.to_dict()).reset_index()
                    nested_col.columns = non_nested_cols + [col_name]
                    nested_df[col_name] = nested_col[col_name]
            # Similar as before, we need to iteratively add the values to account for uneven group sizes
            else:
                for col_name, col_select in cols.items():
                    nested_df[col_name] = 0
                    for index, row in nested_df.iterrows():
                        group = row[non_nested_cols].values
                        filtered_data = data[(data[non_nested_cols] == group).values]
                        nested_df.loc[index, col_name] = [filtered_data[col_select].to_dict()]
        nested_df.index = np.arange(len(nested_df))
    else:
        pass
    return nested_df


def unnest(data, cols, keep_empty=False, ptype=None, names_sep=None, names_repair="check_unique"):
    """Unnesting flattens a column of dicts back out into regular columns

    Parameters
    ----------
    data: pandas or pyspark column
        A data frame
    cols: str or list
        Names of columns to unnest. If you unnest() multiple columns, parallel entries must be of compatible sizes, i.e.
        they're either equal or length 1 (following the standard tidyverse recycling rules).
    keep_empty: bool, default is False
        By default, you get one row of output for each element of the list you're nnesting. This means that
        if there's a size-0 element (like NA or an empty dictionary), that entire row will be dropped from the output.
        If you want to preserve all rows, use keep_empty=True to replace size-0 elements with a single row of missing
        values.
    ptype: dict, default is None
        Optionally, supply a dict for the output cols, overriding the default that will be guessed from the combination
        of individual values.
    names_sep: str, default is None
        If None, the default, the names of new columns will come directly from the inner data frame.
        If a string, the names of the new columns will be formed by pasting together the outer column name with the
        inner names, separated by names_sep.
    names_repair: str, default is "check_unique"
        Used to check that output data frame has valid names. Must be one of the following options:
            "minimal": no name repair or checks, beyond basic existence,
            "unique": make sure names are unique and not empty,
            "check_unique": (the default), no name repair, but check they are unique,

    Returns
    -------
    Our dataframe, but with unnested columns
    """
    is_pandas = _check_df_type(data, "unnest")
    if isinstance(cols, str):
        cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, (tuple, list, (np.ndarray, np.generic))):
        cols = _get_list_columns(data, cols, is_pandas)
    else:
        raise TypeError("Cannot determine columns for unnesting")
    if is_pandas:
        unnested_df = pd.DataFrame()
        non_nested_cols = data.columns.difference(cols).values
        for col in cols:
            exploded_df = pd.DataFrame()
            for index, row in data.iterrows():
                temp_df = pd.DataFrame(data[col][index])
                # Rename the columns based on the outer columns, the separator and the inner column
                if names_sep is not None:
                    temp_df.columns = ['{}{}{}'.format(col, names_sep, val) for val in temp_df.columns]
                # If we haven't concatenated any data yet, we want to include our non_nested_columns at the beginning
                # then we only need to concatenate regular values.
                if len(unnested_df) == 0:
                    for nest_col in non_nested_cols:
                        temp_df[nest_col] = data.loc[index, nest_col]
                    # Re-order the dataset so that the non-nested columns appear first rather than last (or the middle)
                    temp_df = temp_df[non_nested_cols.tolist() + list(temp_df.columns.difference(non_nested_cols))]
                # Add all the temporary datasets row-wise
                exploded_df = pd.concat([exploded_df, temp_df], axis=0)
            # Add all the exploded datasets column wise
            unnested_df = pd.concat([unnested_df, exploded_df], axis=1)
        unnested_df = _check_unique(unnested_df, names_repair)
        if ptype is not None:
            unnested_df = unnested_df.astype(ptype)
        if keep_empty is False:
            unnested_df = unnested_df.dropna()
    else:
        ...
    return unnested_df


# Chopping and Unchopping Data

def chop(data, cols):
    """Makes dataframes shorter by converting rows within each group into lists

    Parameters
    ----------
    data: pandas or pyspark column
        A data frame.
    cols: str or list
        Column to chop. This should be a list-column containing generalised vectors (e.g. any mix of list, numpy array,
        a tuple, or data frames).

    Returns
    -------
    Our dataframe, but with "chopped" rows
    """
    is_pandas = _check_df_type(data, "chop")
    if isinstance(cols, str):
        cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, (tuple, list, (np.ndarray, np.generic))):
        cols = _get_list_columns(data, cols, is_pandas)
    else:
        raise ValueError("Cannot determine columns names for chopping")
    if is_pandas:
        # Get all columns that won't be "chopped"
        unchopped_cols = data.columns.difference(cols).values.tolist()
        # Get all unique unchopped_cols combinations
        chopped_df = data[unchopped_cols].drop_duplicates()
        # Reset the index for concatenation
        chopped_df.index = np.arange(len(chopped_df))
        for col in cols:
            # Here, we transform all the values into a list, reset the index so that they match for concatenation, and
            # then remove the unchopped columns to prevent duplicate column names
            list_cols = data.groupby(unchopped_cols)[col].apply(lambda x: x.values.tolist()).reset_index().drop(unchopped_cols, axis=1)
            # We concatenate every dataframe by column axis
            chopped_df = pd.concat([chopped_df, list_cols], axis=1)
    else:
        ...
    return chopped_df


def unchop(data, cols, keep_empty=False, ptype=None):
    """unchop() makes df longer by expanding list-columns so that each element of the list-column gets its own row in
    the output.

    Parameters
    ----------
    data: pandas or pyspark column
        A data frame
    cols: str or list
        Column to unchop.This should be a list-column containing generalised vectors (e.g. any mix of list, numpy array,
        a tuple, or data frames).
    keep_empty: bool, default is False
        By default, you get one row of output for each element of the list you're unchopping.
        This means that if there's a size-0 element (like NA or an empty data frame), that entire row will be dropped
        from the output. If you want to preserve all rows, use keep_empty=True to replace size-0 elements with a single
        row of missing values.
    ptype: dict, default is None
        Optionally, supply a dictionary for the output cols, overriding the default that will be guessed from
        the combination of individual values.

    Returns
    -------
    Our dataframe, but with the values unchopped
    """
    is_pandas = _check_df_type(data, "unchop")
    if isinstance(cols, str):
        cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, (tuple, list, (np.ndarray, np.generic))):
        cols = _get_list_columns(data, cols, is_pandas)
    else:
        raise ValueError("Cannot determine columns names for unchopping")
    if is_pandas:
        if len(cols) == 1:
            # Explode only works if the columns are in string or tuple format, so we convert the list format to string
            unchopped_df = data.explode(cols[0])
        else:
            unchopped_cols = data.columns.difference(cols).values
            # This method is supposedly faster than others for exploding multiple columns, need to see it in action
            # to verify
            unchopped_df = data.set_index(unchopped_cols).apply(pd.Series.explode).reset_index()
        if not keep_empty:
            unchopped_df = unchopped_df.dropna(axis=0)
            # The issue with using dropna() is that any column with NA gets automatically converted to object type.
            # However, that means that any columns that are integer or float also get converted to object, even after
            # the NAs have been removed. So we peek ahead, see if it's possible to convert the column to numeric, and
            # if so, convert it.
            for c in cols:
                if pd.to_numeric(unchopped_df[c], errors='coerce').notnull().all():
                    unchopped_df[c] = pd.to_numeric(unchopped_df[c], errors='coerce')
        if ptype is not None:
            unchopped_df = unchopped_df.astype(ptype)
        unchopped_df.index = np.arange(len(unchopped_df))
    else:
        ...
    return unchopped_df


# Splitting and Combining Character Columns


def separate(data, col, into, sep="_", remove=True, convert=False, extra="warn", fill="warn"):
    """Given either regular expression or a vector of character positions, separate() turns a single character column
    into multiple columns.

    Parameters
    ----------
    data: pandas or pyspark column
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
            splits = splits.apply(lambda d: d if isinstance(d, (list, tuple, (np.ndarray, np.generic))) else [])
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
                    warnings.warn("Expected {} pieces. Additional piece discarded in {} rows {}".format(len(into), diff,[i for i in range(splits.shape[0], splits.shape[0] + diff)]))
            splits.columns = into + ['NA'] * diff
            # We do this instead of checking if diff > 0 because the user could've specified certain columns to be
            # removed within into() by calling them "NA"
            if 'NA' in splits.columns:
                splits = splits.drop(["NA"], axis=1)
        elif isinstance(sep, (list, tuple, (np.ndarray, np.generic))):
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
    data: pandas or pyspark column
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
    data: pandas or pyspark column
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
    if not isinstance(input_cols, (list, tuple, (np.ndarray, np.generic))):
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
    data: pandas or pyspark column
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
        elif isinstance(cols, (tuple, list, (np.ndarray, np.generic))):
            cols_to_consider = _get_list_columns(data, cols, is_pandas)
        else:
            raise ValueError("Cannot determine which columns to unite")
        if is_pandas:
            data = data.dropna(axis=0, how='any', subset=cols_to_consider)
        else:
            data = data.dropna(how='any', subset=cols_to_consider)
    return data


def replace_na(data, replace):
    """Replace missing values.

    Parameters
    ----------
    data: pandas or pyspark column
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
    data: pandas or pyspark column
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
    elif isinstance(cols, (tuple, list, (np.ndarray, np.generic))):
        cols_to_consider = _get_list_columns(data, cols, is_pandas=is_pandas)
    else:
        raise ValueError("Cannot determine which columns to fill")
    # For reference:
    #  ffill: propagate last valid observation forward to next valid
    #  bfill: use next valid observation to fill gap.
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
    data: pandas or pyspark column
        A data frame
    cols: list, default is None
        Specification of columns to expand
    fill: dictionary, default is None
        A dictionary that for each variable supplies a single value to use instead of NA for missing combinations.

    Returns
    -------
    Our dataframe, but with implicit missing values now made explicit
    """
    is_pandas = _check_df_type(data, "complete")
    # For nesting, we need to compute the possible combinations within each group, as pandas default is to find
    # every possible combination for every column, which isn't always the behavior we want (hence, nesting).
    # For example, if we have complete(data, "group, nesting(item_name, item_id)"), then we are interested in seeing
    # all possible item_id and item_name permutations per group.
    if is_pandas:
        ordered_cols = data.columns.tolist()
        if any('nesting(' in mystring for mystring in cols):
            # These are columns that we aren't interested in seeing all possible permutations of, but rather the ones
            # that act as our consistent pivot. In the example above, it would be group.
            index_cols = [col for col in cols if 'nesting(' not in col]
            index_cols = _get_list_columns(data, index_cols, is_pandas)
            # These are the columns that we are interested in seeing all possible permutations of. In the example above,
            # it would be item_id and item_name
            # Here, we take the column that has nesting in it. Unfortunately, we can't use list.index('nesting(') as it
            # looks for exact matches and we are looking for a substring. We then turn it into a string so we can apply
            # re.search on it.
            nesting_col = [col for col in cols if 'nesting(' in col][0]
            # We then split nesting_col into all of its components. In the example above, it would be 'item_id' and
            # 'item_name'.
            nested_cols = re.search(r'\((.*?)\)', nesting_col).group(1).split(',')
            # Our splitting criteria can cause empty whitespace to occur between strings, so we replace it with
            # non-whitespace.
            nested_cols = [col.replace(' ', '') for col in nested_cols]
            # We make sure that the columns follow our guidelines.
            nested_cols = _get_list_columns(data, nested_cols, is_pandas)
            # This is going to be our dataframe that contains all reindexed/explicit values.
            nested_df = pd.DataFrame()
            # These are the columns that will have missing values as we convert them from implicitly missing to
            # explicitly missing
            value_cols = data.columns.difference(index_cols + nested_cols).values.tolist()
            # Here, we iteratively go through each nesting_col and reindex it. We then merge it with the previously
            # created data frames to get our final result.
            for col in nested_cols:
                group_cols = [col] + index_cols
                group_data = data.set_index(group_cols)
                # We rely on MultiIndex.from_product() to get all possible permutations, and set the names equal to the
                # columns we assigned
                mux = pd.MultiIndex.from_product([group_data.index.levels[i] for i in range(len(group_cols))], names=group_cols)
                # We reindex our data to turn the implicit values explicit, and then reset_index() so that our indexes
                # become columns again
                group_data = group_data.reindex(mux).reset_index()
                # We remove any duplicated columns that aren't going to be used for left joins, because otherwise we
                # are going to get a lot of columns like group_x, group_y, group_z and it becomes very difficult to
                # know which one(s) to remove and which one(s) to keep.
                group_data = group_data.drop([c for c in nested_cols if c != col], axis=1)
                # If we haven't added anything to our nested_df, we can't merge on it (as it has no column to merge onto)
                # so we add our existing data to it.
                if len(nested_df) == 0:
                    nested_df = pd.concat([nested_df, group_data], axis=1)
                # We rely on left joins to add new columns to our dataset
                else:
                    nested_df = pd.merge(nested_df, group_data, on=index_cols + value_cols, how='left')
            data = nested_df.sort_values(index_cols).reset_index()
        # This one is much easier, as no nesting is required. Here, we apply everything as done above, but with no
        # worry about grouping or nesting.
        else:
            data = data.set_index(cols)
            mux = pd.MultiIndex.from_product([data.index.levels[i] for i in range(len(cols))], names=cols)
            data = data.reindex(mux).reset_index()
        if fill is not None:
            data = data.fillna(fill)
        data = data[ordered_cols]
    else:
        ...
        if fill is not None:
            data = data.fillna(fill)
    return data
