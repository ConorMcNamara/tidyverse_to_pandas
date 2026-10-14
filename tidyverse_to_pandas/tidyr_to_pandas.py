"""Convert tidyr data-reshaping syntax to pandas equivalents."""

import re
import warnings

import numpy as np
import pandas as pd

from tidyverse_to_pandas._optional_pyspark import concat_ws, ps
from tidyverse_to_pandas.utils import (
    _check_df_type,
    _check_unique,
    _convert_numeric,
    _get_list_columns,
    _get_str_columns,
)


def _is_list_like(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _is_missing_scalar(x) -> bool:
    return x is None or (not _is_list_like(x) and not isinstance(x, (dict, pd.DataFrame)) and pd.isna(x))


def _place_columns(data: pd.DataFrame, new_data: pd.DataFrame, anchor: str, remove: bool) -> pd.DataFrame:
    """Insert the columns of new_data next to anchor (replacing it if remove), keeping data's row order."""
    new_data = new_data.set_axis(data.index, axis=0)
    position = data.columns.get_loc(anchor)
    left = data.iloc[:, :position] if remove else data.iloc[:, : position + 1]
    right = data.iloc[:, position + 1 :]
    return pd.concat([left, new_data, right], axis=1)


# Pivoting


def pivot_longer(
    data: pd.DataFrame | ps.DataFrame,
    cols: list | tuple | np.ndarray | str,
    names_to: str = "name",
    names_prefix: list | tuple | np.ndarray | str | None = None,
    names_sep: list | tuple | np.ndarray | str | None = None,
    names_pattern: list | tuple | np.ndarray | str | None = None,
    names_ptypes: list | tuple | np.ndarray = None,
    names_repair: str = "check_unique",
    values_to: str = "value",
    values_drop_na: bool = False,
    values_ptypes: [list, tuple, np.ndarray] = None,
) -> pd.DataFrame | ps.DataFrame:
    """Pivot_longer() "lengthens" data, increasing the number of rows and decreasing the number of columns.

    Parameters
    ----------
    data : pandas or pyspark data frame
        A data frame to pivot.
    cols : str or list
        Columns to pivot into longer format.
    names_to : str or list, default is "name"
        A string specifying the name of the column to create from the data stored in the column names of data.
        Can be a list, creating multiple columns, if names_sep or names_pattern is provided.
    names_prefix : str, default is None
        A regular expression used to remove matching text from the start of each variable name.
    names_sep : list or str, default is None
        If names_to contains multiple values, these arguments control how the column name is broken up.
        names_sep takes the same specification as separate(), and can either be a numeric vector (specifying positions
        to break on), or a single string (specifying a regular expression to split on).
    names_pattern : list or str, default is None
        If names_to contains multiple values, these arguments control how the column name is broken up.
        names_pattern takes the same specification as extract(), a regular expression containing matching groups (()).
    names_ptypes : list, default is None
        A list of column name-prototype pairs. A prototype (or ptype for short) is a zero-length vector
        (like integer() or numeric()) that defines the type, class, and attributes of a vector.

        If not specified, the type of the columns generated from names_to will be character.
    names_repair : str, default is "check_unique"
        What happen if the output has invalid column names?
        The default, "check_unique" is to error if the columns are duplicated.
        Use "minimal" to allow duplicates in the output, or "unique" to de-duplicated by adding numeric suffixes.
    values_to : str, default is None
        A string specifying the name of the column to create from the data stored in cell values.
        If names_to is None this value will be ignored, and the name of the value column will be "value".
    values_drop_na : bool, default is False
        If True, will drop rows that contain NAs in the values_to column. This effectively converts explicit missing values
        to implicit missing values, and should generally be used only when missing values in data were created by its structure.
    values_ptypes : list, default is None
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
        # Multiple columns to pivot/melt on
        elif isinstance(cols, (list, tuple, (np.ndarray, np.generic))):
            cols = _get_list_columns(data, cols, is_pandas)
        else:
            raise TypeError("Cannot determine method for determining column types")
        # Columns we aren't pivoting on keep their original order, as in tidyr
        id_vars = [c for c in data.columns if c not in cols]
        names_list = [names_to] if isinstance(names_to, str) else list(names_to)
        # Melt into a temporary name column so that the prefix can be stripped before any splitting happens
        name_col = "__pivot_longer_name__"
        melted_data = pd.melt(
            data.reset_index(drop=True),
            id_vars=id_vars,
            value_vars=cols,
            var_name=name_col,
            value_name=values_to,
        )
        # pd.melt() stacks column by column, whereas tidyr returns every pivoted column for row 1, then row 2, etc.
        row_position = np.tile(np.arange(len(data)), len(cols))
        melted_data = melted_data.iloc[np.argsort(row_position, kind="stable")]
        # Drop rows whose value is missing; missing values in other columns are left alone
        if values_drop_na:
            melted_data = melted_data.dropna(subset=[values_to])
        # names_prefix is a regular expression anchored to the start of each name
        if names_prefix is not None:
            melted_data[name_col] = melted_data[name_col].str.replace(f"^(?:{names_prefix})", "", regex=True)
        if isinstance(names_to, str):
            melted_data = melted_data.rename(columns={name_col: names_to})
        # Uses similar argumentation to separate
        elif names_sep is not None:
            melted_data = separate(
                melted_data,
                name_col,
                names_list,
                names_sep,
                remove=True,
                convert=False,
                extra="drop",
                fill="right",
            )
        # Uses similar arguments to extract
        elif names_pattern is not None:
            melted_data = extract(
                melted_data,
                name_col,
                names_list,
                names_pattern,
                remove=True,
                convert=False,
            )
        else:
            raise ValueError("Cannot determine how to separate into multiple columns")
        # tidyr places the new name columns before the value column
        melted_data = melted_data[id_vars + [n for n in names_list if n in melted_data.columns] + [values_to]]
        melted_data = melted_data.reset_index(drop=True)
        # Check for repeated names
        melted_data = _check_unique(melted_data, how=names_repair)
        # Convert pytypes
        if names_ptypes is not None:
            if isinstance(names_ptypes, dict):
                conversion_type = names_ptypes
            else:
                if not isinstance(names_ptypes, (list, tuple, np.ndarray)):
                    names_ptypes = [names_ptypes]
                conversion_type = dict(zip(names_list, names_ptypes))
            melted_data = melted_data.astype(conversion_type)
        if values_ptypes is not None:
            melted_data[values_to] = melted_data[values_to].astype(values_ptypes)
    else:
        pass
    return melted_data


def pivot_wider(
    data: pd.DataFrame | ps.DataFrame,
    id_cols: str | list | tuple | np.ndarray = None,
    names_from: str = "name",
    names_prefix: str = "",
    names_sep: str = "_",
    names_repair: str = "check_unique",
    values_from: str | list | tuple | np.ndarray = "value",
    values_fill: dict | None = None,
    values_fn: dict | None = None,
) -> pd.DataFrame | ps.DataFrame:
    """Pivot_wider() "widens" data, increasing the number of columns and decreasing the number of rows.

    The inverse transformation is pivot_longer().

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame to pivot.
    id_cols : str or list, default is None
        A set of columns that uniquely identifies each observation.
        Defaults to all columns in data except for the columns specified in names_from and values_from.
        Typically used when you have additional variables that is directly related.
    names_from : str, default is "name"
        Describes which column to get the name of the output column.
    names_prefix : str, default is ""
        String added to the start of every variable name. This is particularly useful if names_from is a numeric vector
        and you want to create syntactic variable names.
    names_sep : str, default is "_"
        If names_from or values_from contains multiple variables, this will be used to join their values together
        into a single string to use as a column name.
    names_repair : str, default is "check_unique"
        What happen if the output has invalid column names? The default, "check_unique" is to error if the columns are duplicated.
        Use "minimal" to allow duplicates in the output, or "unique" to de-duplicated by adding numeric suffixes.
    values_from : str or list, default is "name"
        Describes which column (or columns) to get the cell values from.
    values_fill : dict, default is None
        Optionally, a dictionary specifying what each value should be filled in with when missing.
    values_fn : dict, default is None
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
        names_sep = names_sep if names_sep is not None else "_"
        names_prefix = names_prefix if names_prefix is not None else ""
        if id_cols is None:
            # Keep the id columns in their original order
            id_cols = [c for c in data.columns if c not in list(names_from) + list(values_from)]
        elif isinstance(id_cols, str):
            id_cols = _get_str_columns(data, id_cols, is_pandas=is_pandas)
        else:
            id_cols = _get_list_columns(data, id_cols, is_pandas)
        data = data[list(id_cols) + list(names_from) + list(values_from)]
        # Unites all naming columns into one larger column
        if len(names_from) > 1:
            new_col = names_sep.join(names_from)
            data = unite(data, new_col, input_cols=names_from, sep=names_sep, remove=True, na_rm=False)
            names_from = [new_col]
        names_col = names_from[0]
        # With no id columns, tidyr returns a single row
        id_keys = list(id_cols)
        if not id_keys:
            id_keys = ["__pivot_wider_row__"]
            data = data.assign(**{id_keys[0]: 0})
        grouped = data.groupby(id_keys + [names_col], sort=False, dropna=False)[values_from]
        if values_fn is not None:
            fns = values_fn if isinstance(values_fn, dict) else {v: values_fn for v in values_from}
            long_data = grouped.agg({v: fns.get(v, "first") for v in values_from})
        elif (grouped.size() > 1).any():
            # Like tidyr, keep every duplicated value (as a list) rather than silently summarising them
            warnings.warn(
                "Values are not uniquely identified; output will contain lists. Use values_fn to summarise duplicates."
            )
            long_data = grouped.agg(list)
        else:
            long_data = grouped.first()
        pivoted_data = long_data.unstack(names_col)
        # unstack() sorts; tidyr keeps rows and new columns in the order they first appear
        name_order = pd.unique(data[names_col])
        pivoted_data = pivoted_data.reindex(columns=pd.MultiIndex.from_product([values_from, name_order]))
        row_order = data[id_keys].drop_duplicates()
        if len(id_keys) > 1:
            pivoted_data = pivoted_data.reindex(pd.MultiIndex.from_frame(row_order))
        else:
            pivoted_data = pivoted_data.reindex(pd.Index(row_order[id_keys[0]]))
        # Flatten the column names: "{prefix}{name}", or "{value}{sep}{prefix}{name}" for multiple value columns
        new_names, value_of = [], {}
        for value_col, name in pivoted_data.columns:
            name = "NA" if pd.isna(name) else str(name)
            flat = f"{names_prefix}{name}" if len(values_from) == 1 else f"{value_col}{names_sep}{names_prefix}{name}"
            new_names.append(flat)
            value_of[flat] = value_col
        pivoted_data.columns = new_names
        # Filling missing values
        if values_fill is not None:
            for column in new_names:
                fill_value = values_fill.get(value_of[column]) if isinstance(values_fill, dict) else values_fill
                if fill_value is not None:
                    pivoted_data[column] = pivoted_data[column].fillna(fill_value)
        # Adding our id_cols back as variables instead of indexes
        pivoted_data = pivoted_data.reset_index()
        if not id_cols:
            pivoted_data = pivoted_data.drop(columns=id_keys)
        pivoted_data.columns.name = None
        # Check for repeated names
        pivoted_data = _check_unique(pivoted_data, how=names_repair)
    else:
        pass
    return pivoted_data


# Rectangling Data


def unnest_longer(
    data: pd.DataFrame | ps.DataFrame,
    col: str | list | tuple | np.ndarray,
    values_to: str | list | tuple | np.ndarray = None,
    indices_to: str = None,
    indices_include: bool = False,
    names_repair: str = "check_unique",
    simplify: bool = False,
    ptype: dict | None = None,
) -> pd.DataFrame | ps.DataFrame:
    """Turn each element of a list-column into a row.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    col : str or list
        Column to extract components from
    values_to : str or list, default is None
        Name of column to store vector values. If None, use col
    indices_to : str, default is None
        A string giving the name of column which will contain the inner names or position (if not named) of the values.
        If None, use col with _id suffix
    indices_include : bool, default is False
        Add an index column? Defaults to False when col has inner names.
    names_repair : str, default is check_unique
        "minimal": no name repair or checks, beyond basic existence,
        "unique": make sure names are unique and not empty,
        "check_unique": (the default), no name repair, but check they are unique
    simplify : bool, default is False
        If True, will attempt to simplify lists of length-1 vectors to an atomic vector
    ptype : dict, default is None
        Optionally, supply a dictionary for the output cols, overriding the default that will be guessed from
        the combination of individual values.

    Returns
    -------
    Our dataset, but with the nested columns expanded row-wise.
    """
    is_pandas = _check_df_type(data, "unnest_longer")
    if is_pandas:
        if isinstance(col, str):
            cols = [col]
        elif isinstance(col, (tuple, list, (np.ndarray, np.generic))):
            cols = list(col)
        else:
            raise TypeError("Cannot determine columns to unnest dataframe")

        def _pairs(x) -> list:
            # (index, value) pairs: names for dicts, 1-based positions for lists, a single value otherwise
            if isinstance(x, dict):
                return list(x.items())
            if _is_list_like(x):
                return list(enumerate(x, start=1))
            return [(1, x)]

        def _name(template, c: str, default: str) -> str:
            if template is None:
                return default
            if isinstance(template, str):
                return template.replace("{col}", c) if "{col}" in template or len(cols) > 1 else template
            return list(template)[cols.index(c)]

        unnest_data = data.copy()
        explode_cols, renames = [], {}
        for c in cols:
            pairs = data[c].apply(_pairs)
            is_named = data[c].apply(lambda x: isinstance(x, dict)).any()
            unnest_data[c] = pairs.apply(lambda p: [v for _, v in p])
            explode_cols.append(c)
            renames[c] = _name(values_to, c, c)
            if indices_include or is_named:
                id_col = _name(indices_to, c, f"{c}_id")
                unnest_data.insert(unnest_data.columns.get_loc(c) + 1, id_col, pairs.apply(lambda p: [k for k, _ in p]))
                explode_cols.append(id_col)
        # Size-0 elements (empty lists/dicts) produce no rows, as in tidyr
        is_empty = unnest_data[cols].apply(lambda column: column.apply(len) == 0).all(axis=1)
        unnest_data = unnest_data[~is_empty].explode(explode_cols)
        unnest_data[explode_cols] = unnest_data[explode_cols].infer_objects()
        unnest_data = unnest_data.rename(columns=renames)
        if simplify:
            for c in renames.values():
                try:
                    unnest_data[c] = pd.to_numeric(unnest_data[c])
                except (ValueError, TypeError):
                    pass
        if ptype is not None:
            unnest_data = unnest_data.astype(ptype)
        unnest_data = _check_unique(unnest_data, how=names_repair)
        unnest_data = unnest_data.reset_index(drop=True)
    else:
        ...
    return unnest_data


def unnest_wider(
    data: pd.DataFrame | ps.DataFrame,
    col: str | list | tuple | np.ndarray,
    names_sep: str | None = None,
    simplify: bool = False,
    names_repair: str = "check_unique",
    ptype: dict | None = None,
) -> pd.DataFrame | ps.DataFrame:
    """Turn each element of a list-column into a column.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame.
    col : str or list
        List-column to extract components from.
    names_sep : str, default is None
        If None, the default, the names of new columns will come directly from the inner json.
        If a string, the names of the new columns will be formed by pasting together the outer column name with the
         inner names, separated by names_sep.
    simplify : bool, default is False
        If True, will attempt to simplify lists of length-1 vectors to an atomic vector
    names_repair : str, default is check_unique
        "minimal": no name repair or checks, beyond basic existence,
        "unique": make sure names are unique and not empty,
        "check_unique": (the default), no name repair, but check they are unique
    ptype : dict, default is None
        Optionally, supply a dictionary for the output cols, overriding the default that will be guessed from
        the combination of individual values.

    Returns
    -------
    Our dataset, but with our nested columns expanded column-wise
    """
    is_pandas = _check_df_type(data, "unnest_wider")
    if is_pandas:
        if isinstance(col, str):
            cols = [col]
        elif isinstance(col, (list, tuple, (np.ndarray, np.generic))):
            cols = list(col)
        else:
            raise TypeError("Cannot determine which columns to unnest on")
        unnested_data = data
        for c in cols:
            records = []
            for x in data[c]:
                if isinstance(x, dict):
                    keys = x.items()
                    records.append({k if names_sep is None else f"{c}{names_sep}{k}": v for k, v in keys})
                elif _is_list_like(x):
                    # Unnamed elements get 1-based positional names, e.g. "x_1"
                    sep = names_sep if names_sep is not None else "_"
                    records.append({f"{c}{sep}{i}": v for i, v in enumerate(x, start=1)})
                else:
                    records.append({})
            wide = pd.DataFrame(records, index=data.index)
            if simplify:
                for w in wide.columns:
                    try:
                        wide[w] = pd.to_numeric(wide[w])
                    except (ValueError, TypeError):
                        pass
            # The new columns replace the list-column in place
            unnested_data = _place_columns(unnested_data, wide, c, remove=True)
        # Check that the names for the columns follow our conventions
        unnested_data = _check_unique(unnested_data, how=names_repair)
        # Change pytypes if necessary
        if ptype is not None:
            unnested_data = unnested_data.astype(ptype)
    else:
        ...
    return unnested_data


# Nesting and Unnesting Data


def nest(data: pd.DataFrame | ps.DataFrame, cols: str | list | tuple | np.ndarray) -> pd.DataFrame | ps.DataFrame:
    """Nesting transforms multiple rows and columns into nested dictionaries.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    cols : str or list or dict
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
            if (
                data.groupby(non_nested_cols).agg("count").iloc[0, 0]
                == data.groupby(non_nested_cols).agg("count").iloc[1:, 0]
            ).all():
                nested_df = data.groupby(non_nested_cols).apply(lambda x: x.to_dict()).reset_index()
                nested_df = nested_df.rename({0: "data"}, axis=1)
            # If our groups have different number of observations, then we need to iteratively add their dictionary
            # values as groupby and lambda will only select one value instead of all of them
            else:
                # Ensures that we don't have any unnecessary copies
                nested_df = data[non_nested_cols].drop_duplicates()
                # object dtype so each cell can hold a list of dicts (pandas 3.0 no
                # longer silently upcasts an int column on assignment)
                nested_df["data"] = pd.Series(0, index=nested_df.index, dtype=object)
                for index, row in nested_df.iterrows():
                    group = row[non_nested_cols].values
                    filtered_data = data[(data[non_nested_cols] == group).values]
                    nested_df.at[index, "data"] = filtered_data[cols].to_dict()
        # Cols in dictionary format
        else:
            non_nested_cols = data.columns.difference(nested_cols).tolist()
            nested_df = data[non_nested_cols].drop_duplicates()
            nested_df.index = np.arange(len(nested_df))
            # Similar to above, we are checking if every group in non_nested_cols has the same number of observations to
            # run our groupby and lambda expression successfully
            if (
                data.groupby(non_nested_cols).agg("count").iloc[0, 0]
                == data.groupby(non_nested_cols).agg("count").iloc[1:, 0]
            ).all():
                for col_name, col_select in cols.items():
                    nested_col = data.groupby(non_nested_cols)[col_select].apply(lambda x: x.to_dict()).reset_index()
                    nested_col.columns = non_nested_cols + [col_name]
                    nested_df[col_name] = nested_col[col_name]
            # Similar as before, we need to iteratively add the values to account for uneven group sizes
            else:
                for col_name, col_select in cols.items():
                    nested_df[col_name] = pd.Series(0, index=nested_df.index, dtype=object)
                    for index, row in nested_df.iterrows():
                        group = row[non_nested_cols].values
                        filtered_data = data[(data[non_nested_cols] == group).values]
                        nested_df.at[index, col_name] = filtered_data[col_select].to_dict()
        nested_df.index = np.arange(len(nested_df))
    else:
        pass
    return nested_df


def unnest(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray,
    keep_empty: bool = False,
    ptype: dict | None = None,
    names_sep: str | None = None,
    names_repair: str = "check_unique",
) -> pd.DataFrame | ps.DataFrame:
    """Unnesting flattens a column of dicts back out into regular columns.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    cols : str or list
        Names of columns to unnest. If you unnest() multiple columns, parallel entries must be of compatible sizes, i.e.
        they're either equal or length 1 (following the standard tidyverse recycling rules).
    keep_empty : bool, default is False
        By default, you get one row of output for each element of the list you're unnesting. This means that
        if there's a size-0 element (like NA or an empty dictionary), that entire row will be dropped from the output.
        If you want to preserve all rows, use keep_empty=True to replace size-0 elements with a single row of missing
        values.
    ptype : dict, default is None
        Optionally, supply a dict for the output cols, overriding the default that will be guessed from the combination
        of individual values.
    names_sep : str, default is None
        If None, the default, the names of new columns will come directly from the inner data frame.
        If a string, the names of the new columns will be formed by pasting together the outer column name with the
        inner names, separated by names_sep.
    names_repair : str, default is "check_unique"
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

        def _as_frame(x, c: str) -> pd.DataFrame | None:
            # Convert one nested element to a data frame, or None if it is empty or missing
            if isinstance(x, pd.DataFrame):
                frame = x.reset_index(drop=True)
            elif isinstance(x, dict):
                if not x:
                    return None
                is_scalar = not any(isinstance(v, (dict, list, tuple, np.ndarray, pd.Series)) for v in x.values())
                frame = pd.DataFrame([x]) if is_scalar else pd.DataFrame(x).reset_index(drop=True)
            elif _is_list_like(x):
                if len(x) and all(isinstance(v, dict) for v in x):
                    frame = pd.DataFrame(list(x))
                else:
                    frame = pd.DataFrame({c: list(x)})
            else:
                return None
            if len(frame) == 0:
                return None
            if names_sep is not None:
                frame.columns = [f"{c}{names_sep}{name}" for name in frame.columns]
            return frame

        pieces, inner_names = [], {c: [] for c in cols}
        for index in data.index:
            frames = {c: _as_frame(data.at[index, c], c) for c in cols}
            for c, frame in frames.items():
                if frame is not None:
                    inner_names[c] += [n for n in frame.columns if n not in inner_names[c]]
            n_rows = max([len(f) for f in frames.values() if f is not None], default=0)
            if n_rows == 0:
                if not keep_empty:
                    continue
                n_rows = 1
            parts = []
            for c in data.columns:
                if c in frames:
                    frame = frames[c]
                    if frame is not None:
                        # Size-1 elements are recycled, following the tidyverse recycling rules
                        parts.append(frame if len(frame) == n_rows else frame.loc[[0] * n_rows].reset_index(drop=True))
                else:
                    parts.append(pd.DataFrame({c: [data.at[index, c]] * n_rows}))
            pieces.append(pd.concat(parts, axis=1))
        unnested_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        # Nested columns are expanded in place of the original list-column
        ordered = []
        for c in data.columns:
            ordered += inner_names.get(c, [c])
        unnested_df = unnested_df.reindex(columns=ordered)
        unnested_df = _check_unique(unnested_df, names_repair)
        if ptype is not None:
            unnested_df = unnested_df.astype(ptype)
    else:
        ...
    return unnested_df


# Chopping and Unchopping Data


def chop(data: pd.DataFrame | ps.DataFrame, cols: str | list | tuple | np.ndarray) -> pd.DataFrame | ps.DataFrame:
    """Make dataframes shorter by converting rows within each group into lists.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame.
    cols : str or list
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
        unchopped_cols = [c for c in data.columns if c not in cols]
        if not unchopped_cols:
            chopped_df = pd.DataFrame({c: [data[c].tolist()] for c in cols})
        else:
            # Groups stay in order of first appearance, and missing keys form their own group
            chopped_df = (
                data.groupby(unchopped_cols, sort=False, dropna=False)[cols].agg(lambda x: x.tolist()).reset_index()
            )
        chopped_df = chopped_df[data.columns.tolist()]
    else:
        ...
    return chopped_df


def unchop(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray,
    keep_empty: bool = False,
    ptype: dict | None = None,
) -> pd.DataFrame | ps.DataFrame:
    """Expand list-columns so that each element of the list-column gets its own row in the output.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    cols : str or list
        Column to unchop.This should be a list-column containing generalised vectors (e.g. any mix of list, numpy array,
        a tuple, or data frames).
    keep_empty : bool, default is False
        By default, you get one row of output for each element of the list you're unchopping.
        This means that if there's a size-0 element (like NA or an empty data frame), that entire row will be dropped
        from the output. If you want to preserve all rows, use keep_empty=True to replace size-0 elements with a single
        row of missing values.
    ptype : dict, default is None
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

        def _is_empty(x) -> bool:
            return x is None or (_is_list_like(x) and len(x) == 0)

        unchopped_df = data
        if not keep_empty:
            # Only size-0 elements are dropped; missing values elsewhere in the row are kept
            is_empty = data[cols].apply(lambda column: column.apply(_is_empty)).all(axis=1)
            unchopped_df = data[~is_empty]
        # Parallel list-columns must have matching lengths in each row
        unchopped_df = unchopped_df.explode(cols)
        unchopped_df[cols] = unchopped_df[cols].infer_objects()
        if ptype is not None:
            unchopped_df = unchopped_df.astype(ptype)
        unchopped_df = unchopped_df.reset_index(drop=True)
    else:
        ...
    return unchopped_df


# Splitting and Combining Character Columns


def separate(
    data: pd.DataFrame | ps.DataFrame,
    col: str,
    into: list | tuple | np.ndarray,
    sep: str | int = "_",
    remove: bool = True,
    convert: bool = False,
    extra: str = "warn",
    fill: str = "warn",
) -> pd.DataFrame | ps.DataFrame:
    """Separate a single character column into multiple columns using a regex or vector of character positions.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    col : str
        Column name or position.
    into : list
        Names of new variables to create as character vector. Use "NA" to omit the variable in the output, e.g.
        ['NA', 'B']
    sep : str or int
        Separator between columns.
        If character, is interpreted as a regular expression. The default value is a regular expression that matches any
        sequence of non-alphanumeric values.
        If list or tuple, interpreted as positions to split at. Positive values start at 0 at the far-left of the string;
        negative value start at -1 at the far-right of the string. The length of sep should be one less than into.
    remove : bool, default is True
        If True, remove input column from output data frame.
    convert : bool, default is False
        If True, will run pd.to_numeric() if applicable on new columns.
        This is useful if the component columns are integer, numeric or logical.
    extra : str, options are {"warn", "drop", "merge"}, default is "warn"
        "warn" (the default): emit a warning and drop extra values.
        "drop": drop any extra values without a warning.
        "merge": only splits at most len(into) - 1 times
    fill : str, options are {"warn", "right", "left"}, default is "warn"
        "warn" (the default): emit a warning and fill from the right
        "right": fill with missing values on the right
        "left": fill with missing values on the left

    Returns
    -------
    Our dataframe, but with separated character columns
    """
    is_pandas = _check_df_type(data, "separate")
    extra, fill = extra.casefold(), fill.casefold()
    into = list(into)
    if is_pandas:
        if isinstance(sep, str):
            # 0 for str.split() means we run as many splits as possible, whereas num_times indicates the maximum
            # number of splits we are allowing
            num_times = len(into) - 1 if extra == "merge" else 0
            is_missing = data[col].isna()
            pieces = data[col].str.split(pat=sep, n=num_times, expand=False)
            pieces = [list(p) if not missing else [] for p, missing in zip(pieces, is_missing)]
            # Row numbers are reported 1-based, as tidyr does
            too_many = [i + 1 for i, p in enumerate(pieces) if len(p) > len(into)]
            too_few = [
                i + 1 for i, (p, missing) in enumerate(zip(pieces, is_missing)) if not missing and len(p) < len(into)
            ]
            if extra == "warn" and too_many:
                warnings.warn(
                    f"Expected {len(into)} pieces. Additional pieces discarded in {len(too_many)} rows {too_many}."
                )
            if fill == "warn" and too_few:
                warnings.warn(
                    f"Expected {len(into)} pieces. Missing pieces filled with `NA` in {len(too_few)} rows {too_few}."
                )
            rows = []
            for p in pieces:
                p = p[: len(into)]
                padding = [np.nan] * (len(into) - len(p))
                rows.append(padding + p if fill == "left" else p + padding)
            splits = pd.DataFrame(rows, columns=range(len(into)), index=data.index)
        elif isinstance(sep, (list, tuple, (np.ndarray, np.generic))):
            # Unlike using character separators, numeric ones do not allow for filling or other error handling, so
            # we need the number of column names to exactly match how many separations we are going to create
            if len(sep) != len(into) - 1:
                raise AttributeError(
                    f"{len(into) - 1} column names were expected but {len(sep)} column names were received"
                )
            bounds = [None] + list(sep) + [None]
            # str.slice() returns "" if there is no match, which will fail our covert condition, so we convert it to
            # NaN instead to represent that there was no match
            splits = pd.concat(
                [
                    data[col].str.slice(start=bounds[i], stop=bounds[i + 1]).replace("", np.nan).rename(i)
                    for i in range(len(into))
                ],
                axis=1,
            )
        else:
            raise ValueError("Cannot determine method of splitting columns")
        # Columns named "NA" (or None) are omitted from the output
        keep = [i for i, name in enumerate(into) if not (name is None or name == "NA")]
        splits = splits[keep]
        splits.columns = [into[i] for i in keep]
        if convert:
            splits = _convert_numeric(splits)
        # The new columns go where the original column was
        data = _place_columns(data, splits, col, remove)
    else:
        # pyspark has no native ability to extract column names, so we'll need to do it ourselves
        if remove:
            data = data.drop(col)
    return data


def extract(
    data: pd.DataFrame | ps.DataFrame,
    col: str,
    into: list | tuple | np.ndarray,
    regex: str = "([a-zA-Z0-9]+)",
    remove: bool = True,
    convert: bool = False,
) -> pd.DataFrame | ps.DataFrame:
    """Given a regular expression with capturing groups, extract() turns each group into a new column.

    If the groups don't match, or the input is NA, the output will be NA.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame.
    col : str
        Column name or position
    into : list
        Names of new variables to create as character vector. Use "NA" to omit the variable in the output, e.g.
        ["NA", "B"]
    regex : str, default is all alpha-numeric characters
        A regular expression used to extract the desired values.
        There should be one capture group (defined by ()) for each element of into.
    remove : bool, default is True
        If True, remove input column from output DataFrame
    convert : bool, default is False
        If True, will run pd.to_numeric() if applicable on new columns.
        This is useful if the component columns are integer, numeric or logical.

    Returns
    -------
    Our dataframe, but with extracted character columns
    """
    is_pandas = _check_df_type(data, "extract")
    into = list(into)
    if is_pandas:
        # Here, we are setting expand=True so that pandas returns a DataFrame instead of a Series of lists
        splits = data[col].str.extract(regex, expand=True)
        # If our regex extracts more columns column names provided, then we are keeping the first n columns,
        # where n is the number of column names provided
        if len(into) < splits.shape[1]:
            splits = splits.iloc[:, 0 : len(into)]
        # Columns named "NA" (or None) are omitted from the output
        keep = [i for i, name in enumerate(into) if not (name is None or name == "NA")]
        splits = splits.iloc[:, keep]
        splits.columns = [into[i] for i in keep]
        if convert:
            splits = _convert_numeric(splits)
        # The new columns go where the original column was
        data = _place_columns(data, splits, col, remove)
    else:
        # use Pyspark regular expressions
        if remove:
            data = data.drop(col)
    return data


def unite(
    data: pd.DataFrame | ps.DataFrame,
    col: str,
    input_cols: list | tuple | np.ndarray = None,
    sep: str = "_",
    remove: bool = True,
    na_rm: bool = False,
) -> pd.DataFrame | ps.DataFrame:
    """Paste together multiple columns into one.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame.
    col : str
        The name of our new column
    input_cols : list, default is None
        A selection of columns. If None, all variables are selected. You can supply bare variable names,
        select all variables between x and z with x:z, or exclude y with -y
    sep : str, default is '_'
        Separator to use between values.
    remove : bool, default is True
        If True, remove input columns from output data frame.
    na_rm : bool, default is False
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

        def _paste(row) -> str:
            # Missing values are written as "NA", or skipped entirely when na_rm is True
            values = [None if pd.isna(v) else str(v) for v in row]
            if na_rm:
                return sep.join(v for v in values if v is not None)
            return sep.join("NA" if v is None else v for v in values)

        united = [_paste(row) for row in data[input_cols].itertuples(index=False)]
        # The new column goes where the first input column was
        position = data.columns.get_loc(input_cols[0])
        if remove:
            position -= sum(c in input_cols for c in data.columns[:position])
            data = data.drop(columns=input_cols)
        else:
            data = data.copy()
        data.insert(position, col, united)
    else:
        # concat_ws already skips nulls, so na_rm needs no extra handling
        data = data.withColumn(col, concat_ws(sep, *input_cols))
        if remove:
            data = data.drop(*[c for c in input_cols if c != col])
    return data


# Dealing with missing data


def drop_na(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray = None,
) -> pd.DataFrame | ps.DataFrame:
    """Drop rows containing missing values.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame.
    cols : list of str or str, default is None
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
            data = data.dropna(axis=0, how="any")
        if not is_pandas:
            data = data.dropna(how="any")
    else:  # Drop any row that contains an NA within certain columns
        if isinstance(cols, str):
            cols_to_consider = _get_str_columns(data, cols, is_pandas=is_pandas)
        elif isinstance(cols, (tuple, list, (np.ndarray, np.generic))):
            cols_to_consider = _get_list_columns(data, cols, is_pandas)
        else:
            raise ValueError("Cannot determine which columns to unite")
        if is_pandas:
            data = data.dropna(axis=0, how="any", subset=cols_to_consider)
        else:
            data = data.dropna(how="any", subset=cols_to_consider)
    return data


def replace_na(data: pd.DataFrame | ps.DataFrame, replace: str | int | dict) -> pd.DataFrame | ps.DataFrame:
    """Replace missing values.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    replace : str or int or dict
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


def fill(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray = None,
    direction: str = "down",
) -> pd.DataFrame | ps.DataFrame:
    """Fill missing values in selected columns using the next or previous entry.

    This is useful in the common output format where values are not repeated, and are only recorded when they change.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    cols : str or list, default is None
        A selection of columns. If None, nothing happens.
        If in string format, expected to either be a singular column, such as "x" or "-y" or in "x:z" format if
        selecting multiple columns. Note that for "x:y", we are assuming every character prior to : is the column name,
        so "x :z" will expect the column to be named "x ".
        If in list format, expected to be a list of strings containing column names, such as ["x", "y"] or ["x", "-y"]
    direction : str, options are {"down", "up", "downup", "updown"}, default is 'down'
        Direction in which to fill missing values. Currently either "down" (the default), "up",
        "downup" (i.e. first down and then up) or "updown" (first up and then down).

    Returns
    -------
    Our dataframe, but with missing values filled. Note that integers get converted to floats in pandas
    """
    direction = direction.casefold()
    if direction not in ["down", "up", "downup", "updown"]:
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
        data = data.copy()
        if direction == "down":
            data[cols_to_consider] = data[cols_to_consider].ffill()
        elif direction == "up":
            data[cols_to_consider] = data[cols_to_consider].bfill()
        elif direction == "downup":
            data[cols_to_consider] = data[cols_to_consider].ffill().bfill()
        else:
            data[cols_to_consider] = data[cols_to_consider].bfill().ffill()
    else:
        # Pyspark has no native support for forwards or backwards filling, so need to create my own function to do it
        ...
    return data


def complete(
    data: pd.DataFrame | ps.DataFrame,
    cols: list | tuple | np.ndarray = None,
    fill: dict | None = None,
) -> pd.DataFrame | ps.DataFrame:
    """Turn implicit missing values into explicit missing values.

    Parameters
    ----------
    data : pandas or pyspark column
        A data frame
    cols : list, default is None
        Specification of columns to expand
    fill : dictionary, default is None
        A dictionary that for each variable supplies a single value to use instead of NA for missing combinations.

    Returns
    -------
    Our dataframe, but with implicit missing values now made explicit
    """
    is_pandas = _check_df_type(data, "complete")
    # For nesting, we only want the combinations of those columns that already appear in the data, as opposed to every
    # possible combination. For example, complete(data, ["group", "nesting(item_id, item_name)"]) crosses every group
    # with every (item_id, item_name) pair that exists in the data.
    if cols is None:
        return data.copy() if is_pandas else data
    if isinstance(cols, str):
        cols = [cols]
    if is_pandas:
        ordered_cols = data.columns.tolist()
        frames, key_cols = [], []
        for spec in cols:
            nesting = re.fullmatch(r"\s*nesting\((.*)\)\s*", spec)
            if nesting:
                nested_cols = _get_list_columns(data, [c.strip() for c in nesting.group(1).split(",")], is_pandas)
                frames.append(data[nested_cols].drop_duplicates().sort_values(nested_cols).reset_index(drop=True))
                key_cols += nested_cols
            else:
                for c in _get_str_columns(data, spec, is_pandas=is_pandas):
                    frames.append(data[[c]].drop_duplicates().sort_values(c).reset_index(drop=True))
                    key_cols.append(c)
        # Cross every expansion together, then join the observed rows back on
        grid = frames[0]
        for frame in frames[1:]:
            grid = grid.merge(frame, how="cross")
        data = grid.merge(data, on=key_cols, how="left")
        if fill is not None:
            data = data.fillna(fill)
        data = data[ordered_cols]
    else:
        if fill is not None:
            data = data.fillna(fill)
    return data
