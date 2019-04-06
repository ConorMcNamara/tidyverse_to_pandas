"""The goal of this is to convert dplyr syntax to pandas"""
import pandas as pd
import re

# Transform Variables


def mutate(data, *args):
    """Converts a string function into an assignment for pandas dataframe

    Parameters
    ----------
    data: pandas DataFrame
        The data frame for which we are creating new variables using mutate()
    *args: str
        The variable functions, for example "new_col = col_1 / col_2"

    Returns
    -------
    data: pandas DataFrame
        The data frame, with the new columns created using mutate
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use mutate on a non-DataFrame")
    for arg in args:
        before_equals = re.search(r'(.*)?\s+=', arg).group(0)
        before_arg = re.sub(r'(=|\s+)', r'', before_equals)
        after_equals = re.search(r'=(.*)', arg).group(0)
        after_equals = re.sub(r'(=|\s+)', r'', after_equals)
        after_equals = re.sub(r'\b([a-zA-Z]+)\b', r'x.\1', after_equals)  # Handles camelCase
        if re.search(r'(_)', after_equals):  # Handles snake_case
            after_equals = re.sub(r'\b([a-zA-Z]+_)', r'x.\1', after_equals)
        if re.search(r'([a-zA-Z]+\d)', after_equals):  # Handles camelCase1
            after_equals = re.sub(r'\b([a-zA-Z]+\d)\b', r'x.\1', after_equals)
        if re.search(r'(^\d)', after_equals):  # Handles 18a
            after_equals = re.sub(r'(^\d)', r'x.\1', after_equals)
        after_arg = 'lambda x: {}'.format(after_equals)
        data = data.assign(**{before_arg: eval(after_arg)})
    return data


def transmute(data, *args):
    """Converts a string function into an assignment for pandas DataFrame

    Parameters
    ----------
    data: pandas DataFrame
        The data frame for which we are creating new variables using transmute()
    *args: str
        The variable functions, for example "new_col = col_1 / col_2"

    Returns
    -------
    data: pandas DataFrame
        The data frame, with only the new columns created using transmute
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use transmute on a non-DataFrame")
    cols_to_keep = []
    for arg in args:
        before_equals = re.search(r'(.*)?\s+=', arg).group(0)
        before_arg = re.sub(r'(=|\s+)', r'', before_equals)
        cols_to_keep.append(before_arg)
        after_equals = re.search(r'=(.*)', arg).group(0)
        after_equals = re.sub(r'(=|\s+)', r'', after_equals)
        after_equals = re.sub(r'\b([a-zA-Z]+)\b', r'x.\1', after_equals)  # Handles camelCase
        if re.search(r'(_)', after_equals):  # Handles snake_case
            after_equals = re.sub(r'\b([a-zA-Z]+_)', r'x.\1', after_equals)
        if re.search(r'([a-zA-Z]+\d)', after_equals):  # Handles camelCase1
            after_equals = re.sub(r'\b([a-zA-Z]+\d)\b', r'x.\1', after_equals)
        if re.search(r'(^\d)', after_equals):  # Handles 18a
            after_equals = re.sub(r'(^\d)', r'x.\1', after_equals)
        after_arg = 'lambda x: {}'.format(after_equals)
        data = data.assign(**{before_arg: eval(after_arg)})
    data = data.drop(data.columns.difference(cols_to_keep), axis=1)
    return data


# Select/Rename Variables


def rename(data, *args):
    """Renames columns using the 'x = x_1' arguments, and keeps all the available columns.

    Parameter
    ---------
    data: pandas DataFrame
        The data frame for which we are renaming current columns
    *args: str
        The columns we are renaming, for example, "x = x_1"

    Returns
    -------
    data: pandas DataFrame
        The DataFrame, but with the columns renamed
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use rename on a non-DataFrame")
    cols_to_rename = {}
    for arg in args:
        before_equals = re.search(r'(.*)?\s+=', arg).group(0)
        before_arg = re.sub(r'(=|\s+)', r'', before_equals)
        after_equals = re.search(r'=(.*)', arg).group(0)
        after_arg = re.sub(r'(=|\s+)', r'', after_equals)
        cols_to_rename[before_arg] = after_arg
    data = data.rename(cols_to_rename, axis=1)
    return data


def starts_with(column_name, match, ignore_case=True):
    if ignore_case:
        return re.search(r'(^{}.*)'.format(match.casefold()), column_name.casefold())
    else:
        return re.search(r'(^{}.*)'.format(match), column_name)


def ends_with(column_name, match, ignore_case=True):
    if ignore_case:
        return re.search(r'({}$)'.format(match.casefold()), column_name.casefold())
    else:
        return re.search(r'({}$)'.format(match), column_name)


def contains(column_name, match, ignore_case=True):
    if ignore_case:
        return re.search(r'({})'.format(match.casefold()), column_name.casefold())
    else:
        return re.search(r'({})'.format(match), column_name)


def select(data, *args):
    """Selects columns based on criteria in args

    Parameter
    ---------
    data: pandas DataFrame
        The data frame for which we are renaming current columns
    *args: str
        The columns we are selecting, for example, "x" or "x_1"

    Returns
    -------
    new_data: pandas DataFrame
        A DataFrame with only the renamed columns
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use rename on a non-DataFrame")
    cols_to_keep = []
    cols_to_drop = []
    for arg in args:
        if 'starts_with' in arg:
            expression = re.search(r'\((.*)\)', arg).group(0)
            expression = re.sub(r'(\(|\))', r'', expression)
            for col in data.columns:
                if starts_with(col, r'(^{}.*)'.format(expression)):
                    if re.search(r'(^-)', arg):
                        if col not in cols_to_drop:
                            cols_to_drop.append(col)
                    else:
                        if col not in cols_to_keep:
                            cols_to_keep.append(col)
        elif 'ends_with' in arg:
            expression = re.search(r'\((.*)\)', arg).group(0)
            expression = re.sub(r'(\(|\))', r'', expression)
            for col in data.columns:
                if ends_with(col, r'({}$)'.format(expression)):
                    if re.search(r'(^-)', arg):
                        if col not in cols_to_drop:
                            cols_to_drop.append(col)
                    else:
                        if col not in cols_to_keep:
                            cols_to_keep.append(col)
        elif 'contains' in arg:
            expression = re.search(r'\((.*)\)', arg).group(0)
            expression = re.sub(r'(\(|\))', r'', expression)
            for col in data.columns:
                if contains(col, r'{}'.format(expression)):
                    if re.search(r'(^-)', arg):
                        if col not in cols_to_drop:
                            cols_to_drop.append(col)
                    else:
                        if col not in cols_to_keep:
                            cols_to_keep.append(col)
        elif 'everything' in arg:
            for col in data.columns:
                if re.search(r'(^-)', arg):
                    if col not in cols_to_drop:
                        cols_to_drop.append(col)
                else:
                    if col not in cols_to_keep:
                        cols_to_keep.append(col)
        elif "num_range" in arg:
            var_name = re.search(r'\((.*),', arg).group(0)
            var_name = re.sub(r'(\(|,)', r'', var_name)
            var_range = re.search(r'\d.*(\d)', arg).group(0)
            var_start_range = int(re.search(r'(^\d)', var_range).group(0))
            var_end_range = int(re.search(r'(\d$)', var_range).group(0))
            for i in range(var_start_range, var_end_range + 1):
                if re.search(r"(^-)", arg):
                    cols_to_drop.append('{}{}'.format(var_name, i))
                else:
                    cols_to_keep.append('{}{}'.format(var_name, i))
        elif "last_col" in arg:
            if re.search(r'\((\d+)\)', arg):
                num_offset = re.search(r'\((\d+)\)', arg).group(0)
                num_offset = int(re.sub(r'(\(|\))', r'', num_offset)) + 1
            elif re.search(r'\((offset?\s=?\s\d+)\)', arg):
                num_offset = re.search(r'\((offset?\s=?\s\d+)\)', arg).group(0)
                num_offset = int(re.search(r'\d+', re.sub(r'(\(|\)|\s+)', r'', num_offset)).group(0)) + 1
            elif re.search(r'\((offset=\d+)\)', arg):
                num_offset = re.search(r'\((offset=\d+)\)', arg).group(0)
                num_offset = int(re.search(r'\d+', re.sub(r'(\(|\))', r'', num_offset)).group(0)) + 1
            else:
                num_offset = 1
            if num_offset > len(data.columns):
                raise Exception("Number of offsets is greater than number of columns")
            col = data.iloc[:, -num_offset].name
            if re.search(r"(^-)", arg):
                if col not in cols_to_drop:
                    cols_to_drop.append(col)
            else:
                if col not in cols_to_keep:
                    cols_to_keep.append(col)
        else:
            if re.search(r'(^-)', arg):
                if "-" in re.sub(r'(-?\s+)', r'', arg):
                    if re.sub(r'-', r'', arg) not in cols_to_drop:
                        cols_to_drop.append(re.sub(r'-', r'', arg))
                else:
                    if re.sub(r'(-?\s+)', r'', arg) not in cols_to_drop:
                        cols_to_drop.append(re.sub(r'(-?\s+)', r'', arg))
            else:
                if arg not in data.columns:
                    raise Exception("Selected column {} not found in data frame".format(arg))
                else:
                    if arg not in cols_to_keep:
                        cols_to_keep.append(arg)
    if cols_to_drop:
        if cols_to_keep:
            new_data = pd.concat([data.drop(cols_to_drop, axis=1), data[cols_to_keep]], axis=1)
        else:
            new_data = data.drop(cols_to_drop, axis=1)
    else:
        new_data = data[cols_to_keep]
    return new_data

# Filter Data


def filter(data, *args):
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
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use filter on non-DataFrame")
    query_result = []
    counter = 1
    args_length = len(args)
    for arg in args:
        if "mean(" in arg.casefold():
            mean_col = re.search(r'(?<=mean\()[a-zA-Z]+', arg).group(0)
            val = data[mean_col].mean()
            comparison = re.search(r'([<>]=?|==)', arg).group(0)
            result = '{} {} {}'.format(mean_col, comparison, val)
        elif "median(" in arg.casefold():
            median_col = re.search(r'(?<=median\()[a-zA-Z]+', arg).group(0)
            val = data[median_col].median()
            comparison = re.search(r'([<>]=?|==)', arg).group(0)
            result = '{} {} {}'.format(median_col, comparison, val)
        elif "min(" in arg.casefold():
            min_col = re.search(r'(?<=min\()[a-zA-Z]+', arg).group(0)
            val = data[min_col].min()
            comparison = re.search(r'([<>]=?|==)', arg).group(0)
            result = '{} {} {}'.format(min_col, comparison, val)
        elif "max(" in arg.casefold():
            max_col = re.search(r'(?<=max\()[a-zA-Z]+', arg).group(0)
            val = data[max_col].max()
            comparison = re.search(r'([<>]=?|==)', arg).group(0)
            result = '{} {} {}'.format(max_col, comparison, val)
        else:
            result = arg
        if counter < args_length:
            result = result + ' & '
        query_result.append(result)
        counter += 1
    return data.query(''.join(query_result))


# Summarise Data

def summarise(data, *args):
    """Summarises data based on arguments from args

    Parameters
    ---------
    data: pandas DataFrame
        The dataframe for which we are trying to summarise the data on
    *args: str
        The columns we are performing summary calculations on, as well as type of summary calculations

    Returns
    -------
    summarised_data: pandas DataFrame
        The dataframe containing our summarised results
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use summarise on a non-DataFrame")
    summarise_data = pd.DataFrame()
    for arg in args:
        if "mean(" in arg.casefold():
            mean_col = re.search(r'(?<=mean\()[a-zA-Z]+', arg).group(0)
            val = data[mean_col].mean()
        elif "median(" in arg.casefold():
            median_col = re.search(r'(?<=median\()[a-zA-Z]+', arg).group(0)
            val = data[median_col].median()
        elif "sd(" in arg.casefold():
            sd_col = re.search(r'(?<=sd\()[a-zA-Z]+', arg).group(0)
            val = data[sd_col].std()
        elif "iqr(" in arg.casefold():
            iqr_col = re.search(r'(?<=iqr\()[a-zA-Z]+', arg).group(0)
            lower_25 = data[iqr_col].quantile(0.25)
            upper_25 = data[iqr_col].quantile(0.75)
            val = upper_25 - lower_25
        elif "mad(" in arg.casefold():
            mad_col = re.search(r'(?<=mad\()[a-zA-Z]+', arg).group(0)
            val = data[mad_col].mad()
        elif "min(" in arg.casefold():
            min_col = re.search(r'(?<=min\()[a-zA-Z]+', arg).group(0)
            val = data[min_col].min()
        elif "max(" in arg.casefold():
            max_col = re.search(r'(?<=max\()[a-zA-Z]+', arg).group(0)
            val = data[max_col].max()
        elif "quantile(" in arg.casefold():
            quantile_col = re.search(r'(?<=quantile\()[a-zA-Z]+', arg).group(0)
            if re.search('probs=', arg):
                quantile_percent = float(re.search(r'(?<=probs\=)\s*\d*\.\d+', arg).group(0))
            else:
                quantile_percent = float(re.search(r'(?<=,)\s*\d*\.\d+', arg).group(0))
            if quantile_percent > 1:
                raise Exception("Cannot have percentile greater than 1")
            val = data[quantile_col].quantile(quantile_percent)
        elif "first(" in arg.casefold():
            first_col = re.search(r'(?<=first\()[a-zA-Z]+', arg).group(0)
            val = data.loc[0, first_col]
        elif "last(" in arg.casefold():
            last_col = re.search(r'(?<=last\()[a-zA-Z]+', arg).group(0)
            val = data.loc[len(data) - 1, last_col]
        elif "nth(" in arg.casefold():
            nth_col = re.search(r'(?<=nth\()[a-zA-Z]+', arg).group(0)
            if re.search('n=', arg):
                if re.search("-", arg):
                    n_number = -int(float(re.search(r'(?<=-)\s*\d+', arg).group(0)))
                else:
                    n_number = int(float(re.search(r'(?<=n\=)\s*\d+', arg).group(0)))
            else:
                if re.search("-", arg):
                    n_number = -int(float(re.search(r'(?<=-)\s*\d+', arg).group(0)))
                else:
                    n_number = int(float(re.search(r'(?<=,)\s*\d+', arg).group(0)))
            if n_number == 0:
                n_number = 1
            if n_number > len(data):
                raise Exception("Cannot access {} element of DataFrame with {} elements".format(n_number, len(data)))
            if n_number < -len(data):
                raise Exception("Cannot access {} element of DataFrame with {} elements").format(len(data) + n_number, len(data))
            if n_number < 0:
                val = data.iloc[n_number, :][nth_col]
            else:
                val = data.iloc[n_number - 1, :][nth_col]
        elif "n()" in arg.casefold():
            val = len(data.dropna(axis=1, how='all'))
        elif 'n_distinct(' in arg.casefold():
            n_distinct_col = re.search(r'(?<=n_distinct\()[a-zA-Z]+', arg).group(0)
            val = data[n_distinct_col].nunique()
        if re.search(r"^[a-zA-Z]+\s*=", arg):
            name = re.search(r"^[a-zA-Z]+", arg).group(0)
        else:
            name = arg
        summarise_data = pd.concat([summarise_data, pd.Series(val, name=name)], axis=1)
    return summarise_data


# Sort data

def arrange(data, *args):
    """Sorts the data based on columns provided, as well as if they're in 'desc()'

    Parameters
    ---------
    data: pandas DataFrame
        The dataframe for which we are trying to sort the data on
    *args: str
        The columns we are sorting the dataframe on

    Returns
    -------
    sorted_data: pandas DataFrame
        The sorted data based on arguments provided in *args
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("Cannot use arrange on a non-DataFrame")
    sorting_cols = []
    ascending_cols = []
    for arg in args:
        if re.search('desc()', arg):
            sorting_cols.append(re.sub(r'desc|\(|\)|\s+', r'', arg))
            ascending_cols.append(False)
        else:
            sorting_cols.append(arg)
            ascending_cols.append(True)
    return data.sort_values(sorting_cols, ascending=ascending_cols).reset_index().drop('index', axis=1)
