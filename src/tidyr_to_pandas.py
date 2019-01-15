"""The goal of this is to convert tidyr syntax to pandas usage"""
import pandas as pd
import warnings


def spread(data, key, values, sep=None):
    """Takes in a column with a key-value pair and "spreads" them across multiple columns.

    Parameters
    ----------
    data: pandas DataFrame
        The DataFrame we are using spread() on
    key: str
        The column name in our key-value pair
    values: str
        The column name(s) in our key-value pair
    sep: str
        If not none, then we name our columns as key-sep-key_value. Else, it's key_value

    Returns
    -------
    The dataframe after we've spread it

    For example, suppose we had a dataframe like
    row	var	            value
    1	Sepal.length    5.1
    1	Species         setosa
    1	species_num     1
    51	Sepal.length	7
    51	Species	        versicolor
    51	species_num	    2

    Then after running spread(data, 'var', 'value'), we'll have
    row	Sepal.length	Species	    species_num
    1	5.1	           setosa	    1
    51	7	           versicolor	2
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use spread on non-DataFrame")
    if isinstance(values, str):
        values = [values]
    index_col = data.columns.difference(values + [key])[0]
    spread_data = data.pivot(index=index_col, columns=key, values=values)
    spread_data.columns.name = None
    spread_data = spread_data.reset_index()
    spread_data.columns = [spread_data.columns[0][0]] + [a for a in spread_data.columns.get_level_values(1)[1:]]
    if sep is not None:
        rename_cols = ['{}'.format(index_col)] + ['{}{}{}'.format(key, sep, col) for col in spread_data.columns[range(1, spread_data.shape[1])]]
        spread_data.columns = rename_cols
    return spread_data


def gather(data, key, value, index_cols, na_rm=False):
    """Takes in multiple columns and "gathers" them into key-value pairs. Turns wide/shallow data long/deep.

    Parameters
    -----------
    data: pandas DataFrame
        The DataFrame we are using gather() on
    key: str
        The name of our 'key' in our key-value pair
    value: str
        The name of our 'value' in our key-value pair
    index_cols: str or list
        The columns that we are 'gathering' to make wide data long
    na_rm: boolean
        Whether or not we drop rows that contain NAs

    Returns
    -------
    gathered_data: pandas DataFrame
        The dataframe after we've gathered it

    For example, suppose we had a data frame like
    names   a   b
    Wilbur	67	56
    Petunia	80	90
    Gregory	64	50

    After performing gather(data, 'Treatment', 'Heart Rate', ['a', 'b']), we'd get
    names	Treatment	Heart Rate
    Wilbur	a	        67
    Petunia	a	        80
    Gregory	a	        64
    Wilbur	b	        56
    Petunia	b	        90
    Gregory	b	        50
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use gather on non-DataFrame")
    index_vars = [col for col in data.columns if col not in index_cols]
    gathered_data = pd.melt(data, index_vars, index_cols, key, value)
    if na_rm:
        gathered_data = gathered_data.dropna()
    return gathered_data


def separate(data, col, into, sep='_', remove=True, extra="warn", fill="warn"):
    """

    Parameters
    ----------
    data: pandas DataFrame
        The DataFrame we are performing extract() on
    col: str
        The column that we wish to apply our separate() function on
    into: str or list
        The name(s) of our column(s) after we've applied our separate() function to generate new column(s)
    sep: str
        The separator that we are using to separate column values.
    remove: boolean
        Whether we wish to remove col from our DataFrame
    extra: str
        If "warn", should len(into) be less than the number of columns after we've done separate(), we will warn the users
        that we are renaming the columns up until the end of into, then dropping the remaining columns.
        If "drop", should len(into) be less than the number of columns after we've done separate(), we will
            rename the columns up until the end of into, then drop the remaining columns.
        If "merge", should len(into) be less than the number of columns after we've done separate(), we will limit the amount
            of separate()/extracts to be len(into) - 1.
    fill: str
        If 'left', should len(into) be greater than the number of columns after we've done separate(),  we will add
        'NAs' to left of our extracted values, wherever there would exist an empty value.
        If 'right', should len(into) be greater than the number of columns after we've done separate(), we will add
        'NAs' to the right of our extracted values, wherever there would exist an empty value.
        If 'warning', we will warn the user that we are performing 'right' fill.

    Returns
    -------
    data: pandas DataFrame
        The DataFrame after we've applied separate() on one of its columns

    For example, suppose we had a Series like
    X
    a
    a b
    a b c
    NA

    Then running separate(df, 'X', ['A', 'B'], sep=' ', remove=True, extra='drop', fill='left') would return
    A	B
    NA	a
    a	b
    a	b
    NA	NA

    Meanwhile, running separate(df, 'X', ['A', 'B'], sep=' ', remove=True, extra='warn', fill='right') would return
    A	B
    a	NA
    a	b
    a	b
    NA	NA

    And running separate(df, 'X', ['A', 'B'], sep=' ', remove=True, extra='merge', fill='warn') would return
    A	B
    a	NA
    a	b
    a	b c
    NA	NA
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use separate on non-DataFrame")
    num_times = len(into) - 1 if extra == 'merge' else 0
    new_cols = data[col].str.split(pat=sep, n=num_times, expand=False)
    if fill.casefold() == 'left':
        for i in range(len(new_cols)):
            if len(new_cols[i]) < len(into):
                num_nas = len(into) - len(new_cols[i])
                new_cols[i] = ['NA'] * num_nas + new_cols[i]
    else:
        if fill.casefold() == 'warn':
            warnings.warn('Filling values from right to left')
        for i in range(len(new_cols)):
            if len(new_cols[i]) < len(into):
                num_nas = len(into) - len(new_cols[i])
                new_cols[i] = new_cols[i] + ['NA'] * num_nas
    new_cols = pd.DataFrame(item for item in new_cols)
    diff = new_cols.shape[1] - len(into) if new_cols.shape[1] - len(into) > 0 else 0
    if extra.casefold() == 'warn':
        if diff > 0:
            warnings.warn("Expected {} pieces. Additional piece discarded in {} rows {}".format(len(into), diff,
                                                                                                [i for i in range(new_cols.shape[0], new_cols.shape[0]+diff)]))
    into = into + ['NA'] * diff
    new_cols.columns = into
    if 'na' in [val.casefold() for val in into]:
        drop_cols = [i for i, val in enumerate(into) if val.casefold() == 'na']
        new_cols = new_cols.drop(new_cols.columns[drop_cols], axis=1)
    data = pd.concat([data, new_cols], axis=1)
    if remove:
        data = data.drop(col, axis=1)
    return data


def unite(data, col, index_cols, sep='_', remove=True):
    """Takes in multiple columns and joins their values together

    Parameters
    ----------
    data: pandas DataFrame
        The DataFrame we are using unite() one
    col: str
        The name of the new column we are creating by uniting multiple columns together
    index_cols: str or list
        The name(s) of the column(s) we are uniting to form a new column
    sep: str
        The type of separator used to distinguish between the united values
    remove: boolean
        Whether we wish to remove the index_cols from our DataFrame

    Returns
    -------
    data: pandas DataFrame
        The data frame after we've united the column(s) together.
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use unite on non-DataFrame")
    if not isinstance(col, str):
        raise Exception("Cannot name column using a non-string")
    data[col] = data[index_cols].apply(lambda x: '{}'.format(sep).join(x.map(str)), axis=1)
    if remove:
        data = data.drop(index_cols, axis=1)
    return data


def extract(data, col, into, regex=r'', remove=True):
    """Creates new columns using regular expression on a designated column

    Parameters
    ----------
    data: pandas DataFrame
        The DataFrame we are performing extract() on
    col: str
        The column that our regular expression will be applied on
    into: str or list
        The name(s) of our column(s) after we've applied our regular expressions to generate new column(s)
    regex: regular expression
        The regular expression we are applying on our column col
    remove: boolean
        Whether we wish to remove col from our DataFrame

    Returns
    -------
    data: pandas DataFrame
        The DataFrame after we've applied our regular expression to the column
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use extract on non-DataFrame")
    cols = data[col].str.extract(regex)
    if len(into) < cols.shape[1]:
        cols = cols.iloc[:, range(into)]
    cols.columns = [name for name in into]
    data = pd.concat([data, cols], axis=1)
    if remove:
        data = data.drop(col, axis=1)
    return data

