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
            # Here, we are removing the des() from our string, but not whitespace as we may have a column like ' col1'
            sorting_cols.append(re.sub(r'desc|\(|\)|\s+', r'', c))
            ascending_cols.append(False)
        else:
            sorting_cols.append(c)
            ascending_cols.append(True)
    if is_pandas:
        # Here, we are resetting the index because R's implementation also resets the index. Thus, we also need to drop
        # our prior index
        return data.sort_values(sorting_cols, ascending=ascending_cols).reset_index().drop('index', axis=1)
    else:
        return data.orderBy(sorting_cols, ascending=ascending_cols)


def count(data, cols, wt=None, sort=False, name=None, drop=True):
    """Count observations by group

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A dataframe
    cols: str or list
         Optional variables to use when determining uniqueness. If there are multiple rows for a given combination of inputs,
         only the first row will be preserved. If None, will use all variables.
    wt: str of list, default is None
         Frequency weights. Can be a variable (or combination of variables) or None. wt is computed once for each unique
         combination of the counted variables.
    sort: bool, default is False
        If True, will show the largest groups at the top.
    name: str, default is None
        The name of the new column in the output. If omitted, it will default to n. If there's already a column called n,
        it will error, and require you to specify the name.
    drop: bool, default is True
        If False will include counts for empty groups (i.e. for levels of factors that don't exist in the data)

    Returns
    -------
    A pandas series with our counts or sums
    """
    is_pandas = _check_df_type(data, "count")
    if isinstance(cols, str):
        distinct_cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, list):
        distinct_cols = _get_list_columns(data, cols, is_pandas=is_pandas)
    else:
        if is_pandas:
            distinct_cols = data.columns
        else:
            distinct_cols = data.schema.names
    if name is None:
        name = 'n'
    if is_pandas:
        # Here, we are treating our groupby columns as factors/categorical data, so we need to convert it. This also
        # allows us to incorporate the drop() command, as otherwise it would serve no purpose.
        data[distinct_cols] = data[distinct_cols].astype('category')
        if wt is None:
            count_df = data.groupby(distinct_cols).count()
        else:
            count_df = data.groupby(distinct_cols)[wt].sum()
        # We don't need multiple columns containing the same counts, so we choose the first one and rename it
        if isinstance(count_df, pd.DataFrame):
            count_df = count_df.rename({count_df.columns[0]: name}, axis=1)
            count_df = count_df.iloc[:, 0]
        # We have one column/Series, so all we need to do is rename it
        else:
            count_df = count_df.rename(name)
        # Drops any column that contains NaN (i.e., categories that don't have any observations)
        if drop:
            count_df = count_df.dropna()
        # Replaces the NaNs (i.e., categories that don't have any observations) with 0
        else:
            count_df = count_df.fillna(0)
        # Since we know that count_df is going to be a Series, we don't need to specify the axis or the column names, as
        # those are already known for a Series
        if sort:
            count_df = count_df.sort_values(ascending=False)
        count_df = count_df.reset_index()
    else:
        ...
    return count_df


def add_count(data, cols, wt=None, sort=False, name=None):
    """Count observations by group

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A dataframe
    cols: str or list
         Optional variables to use when determining uniqueness. If there are multiple rows for a given combination of inputs,
         only the first row will be preserved. If None, will use all variables.
    wt: str of list, default is None
         Frequency weights. Can be a variable (or combination of variables) or None. wt is computed once for each unique
         combination of the counted variables.
    sort: bool, default is False
        If True, will show the largest groups at the top.
    name: str, default is None
        The name of the new column in the output. If omitted, it will default to n. If there's already a column called n,
        it will error, and require you to specify the name.

    Returns
    -------
    Our dataframe, but with an additional column including the sum or count
    """
    is_pandas = _check_df_type(data, "add_count")
    if isinstance(cols, str):
        distinct_cols = _get_str_columns(data, cols, is_pandas=is_pandas)
    elif isinstance(cols, list):
        distinct_cols = _get_list_columns(data, cols, is_pandas=is_pandas)
    else:
        if is_pandas:
            distinct_cols = data.columns
        else:
            distinct_cols = data.schema.names
    if name is None:
        name = 'n'
    if is_pandas:
        if wt is None:
            # While it would be nice to simply have df[name] = df.groupby([col])[col].transform('count'), that only
            # works when we know that col is one column. Otherwise, we will be trying to set one column with 2+ columns,
            # which will throw an error. So we instead create our Series/DataFrame and then perform a check to ensure
            # that we don't try and create a new column with several.
            groupby_data = data.groupby(distinct_cols)[distinct_cols].transform('count')
            if isinstance(groupby_data, pd.Series):
                data[name] = groupby_data
            else:
                data[name] = groupby_data.iloc[:, 0]
        else:
            # We know that wt has to be a single column, so we can afford to automatically create a new column and not
            # have to worry about errors caused by setting a column from several.
            data[name] = data.groupby(distinct_cols)[wt].transform('sum')
        if sort:
            data = data.sort_values(by=name, ascending=False)
    else:
        ...
    return data


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
            # Here, we need to find the distinct values, and then perform a left merge on the distinct values with the
            # remaining columns to ensure that we have all available columns. We then reset the index as that is what
            # R's implementation does.
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
        query_result.append(result)
    if is_pandas:
        return data.query(' & '.join(query_result)).reset_index().drop(['index'], axis=1)
    else:
        return data.filter(' and '.join(query_result))


def mutate(data, cols, keep='all'):
    """

    Parameters
    ----------
    data: pandas or pyspark DataFrame
        A dataframe
    cols: str, list or dict
        Name-value pairs. The name gives the name of the column in the output. The value can be:
            A vector of length 1, which will be recycled to the correct length.
            A vector the same length as the current group (or the whole data frame if ungrouped).
            NULL, to remove the column.
            A data frame or tibble, to create multiple columns in the output.
    keep: str, default is all
        Allows you to control which columns from data are retained in the output:
            "all", the default, retains all variables.
            "used" keeps any variables used to make new variables; it's useful for checking your work as it displays inputs and outputs side-by-side.
            "unused" keeps only existing variables not used to make new variables.
            "none", only keeps grouping keys (like transmute()).

    Returns
    -------

    """
    is_pandas = _check_df_type(data, "mutate")
    if is_pandas:
        if keep.casefold() in ['used', 'unused']:
            keep_cols = []
        elif keep.casefold() == 'none':
            keep_cols = data.columns
        if isinstance(cols, str):
            cols = re.sub(r'\s', r'', cols)
            before_equals = re.search(r'.+?(?=)', cols).group(0)
            after_equals = re.search(r'(?<=\=).*', cols).group(0)
            # Handle camelCase
            after_equals = re.sub(r'\b([a-zA-Z]+)\b', r'df.\1', after_equals)
            # Handle Snake Case
            if re.search(r'(_)', after_equals):
                after_equals = re.sub(r'\b([a-zA-Z]+_)', r'df.\1', after_equals)
            # Handle CamelCase1
            if re.search(r'([a-zA-Z]+\d+)', after_equals):
                after_equals = re.sub(r'\b([a-zA-Z]+\d+)\b', r'df.\1', after_equals)
            # Handle 18a
            if re.search(r'(^\d)', after_equals):
                after_equals = re.sub(r'(^\d)', r'df.\1', after_equals)
            # Turn log manipulations into np.log
            if 'df.log(' in after_equals:
                after_equals = re.sub(r'df.log', r'np.log', after_equals)
            if 'df.log2' in after_equals:
                after_equals = re.sub(r'df.log2', r'np.log2', after_equals)
            if 'df.log10(' in after_equals:
                after_equals = re.sub(r'df.log10', r'np.log10', after_equals)
            # Turn cumulative sum into pandas equivalent
            if 'df.cumsum(' in after_equals:
                after_equals = re.sub(r'cumsum\((.*?)\)', re.search(r'cumsum\((.*?)\)', cols).group(1) + '.cumsum()',
                                      after_equals)
            if 'df.cummean(' in after_equals:
                ...
            if 'df.lag(' in after_equals:
                if ',' in re.search(r'(?<=lag\(df\.)[a-zA-Z0-9_,\s]*', after_equals).group(0):
                    lag_string = re.search(r'(?<=lag\()[a-zA-Z0-9_,\s\.=]*', after_equals).group(0)
                    lag_string = re.sub(r'df\.', '', lag_string)
                    if len(lag_string.split(',')) > 2:
                        lag_string = ','.join(lag_string.split(',')[1:])
                    else:
                        lag_string = ''.join(lag_string.split(',')[1:])
                    if 'n=' in lag_string:
                        lag_string = re.sub(r'n=', 'periods=', lag_string)
                    if 'default=' in lag_string:
                        lag_string = re.sub(r'default', 'fill_value', lag_string)
                    if len(lag_string.split(',')) == 2:
                        if 'default' not in lag_string.split(',')[1]:
                            lag_string = re.sub(r',', ',fill_value=', lag_string)
                    after_equals = re.sub(r',.*', '', re.sub('df\.lag\(', '', after_equals)) + '.shift({})'.format(lag_string)
                else:
                    after_equals = re.sub(r'lag\((.*?)\)', re.search(r'lag\((.*?)\)', cols).group(1) + '.shift()',
                                          after_equals)
            if 'df.lead(' in after_equals:
                if ',' in re.search(r'(?<=lead\(df\.)[a-zA-Z0-9_,\s]*', after_equals).group(0):
                    lead_string = re.search(r'(?<=lead\()[a-zA-Z0-9_,\s\.=]*', after_equals).group(0)
                    lead_string = re.sub(r'df\.', '', lead_string)
                    if len(lead_string.split(',')) > 2:
                        lead_string = ','.join(lead_string.split(',')[1:])
                    else:
                        lead_string = ''.join(lead_string.split(',')[1:])
                    if 'n=' in lead_string:
                        lead_string = re.sub(r'n=', 'periods=-', lead_string)
                    if 'default=' in lead_string:
                        lead_string = re.sub(r'default=', 'fill_value=', lead_string)
                    if len(lead_string.split(',')) >= 2:
                        if 'default' not in lead_string.split(',')[1]:
                            lead_string = re.sub(r',', ',fill_value=', lead_string)
                        if 'n=' not in lead_string.split(',')[0]:
                            lead_array = lead_string.split(',')
                            lead_array[0] = str('periods={}'.format(-int(lead_array[0])))
                            lead_string = ','.join(lead_array)
                    else:
                        lead_string = str('periods={}'.format(-int(lead_string)))
                    after_equals = re.sub(r',.*', '', re.sub('df\.lead\(', '', after_equals)) + '.shift({})'.format(lead_string)
                else:
                    after_equals = re.sub(r'lead\((.*?)\)', re.search(r'lead\((.*?)\)', cols).group(1) + '.shift(-1)',
                                          after_equals)
            after_equals = 'lambda df: {}'.format(after_equals)
            data = data.assign(**{before_equals: eval(after_equals)})
            if keep.casefold() in ['none', 'unused']:
                data = data.drop(keep_cols, axis=1)
            elif keep.casefold() == 'used':
                data = data.drop(data.columns.difference(keep_cols), axis=1)
            else:
                pass
        return data


def transmute(data, cols):
    return mutate(data, cols, keep='None')


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