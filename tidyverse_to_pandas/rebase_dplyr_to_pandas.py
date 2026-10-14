"""Convert dplyr syntax to pandas equivalents (rebased implementation)."""

import ast
import keyword
import re
from typing import Any

import numpy as np
import pandas as pd

from tidyverse_to_pandas._optional_pyspark import ps
from tidyverse_to_pandas.utils import _check_df_type, _get_list_columns, _get_str_columns


def _resolve_columns(data: pd.DataFrame | ps.DataFrame, cols: Any, is_pandas: bool) -> list:
    """Expand tidyselect-style column arguments given as a str, list, tuple or numpy array."""
    if isinstance(cols, str):
        return _get_str_columns(data, cols, is_pandas=is_pandas)
    if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
        return _get_list_columns(data, list(cols), is_pandas)
    raise TypeError("Cannot determine method for determining column types")


def _as_list(args: Any) -> list:
    """Normalise a single expression or a collection of expressions into a list."""
    if isinstance(args, str):
        return [args]
    if isinstance(args, (list, tuple, np.ndarray)):
        return list(args)
    raise TypeError("Expressions must be given as a str or a list of str")


# Expression evaluation ------------------------------------------------------------------------------------------------
# dplyr expressions are written in R syntax. Rather than rewriting them with regular expressions, we tokenize them,
# replace column references with placeholders, translate R operators into Python ones, parse the result with ``ast`` and
# evaluate it against a namespace of columns and R-like functions. Python's ``and``/``or``/``not`` give the same
# precedence as R's ``&``/``|``/``!`` (lower than comparisons), and the AST is then rewritten to use vectorised ops.

_TOKEN_RE = re.compile(
    r"""
    (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    |(?P<backtick>`[^`]+`)
    |(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?)
    |(?P<name>[A-Za-z_][A-Za-z0-9_.]*)
    |(?P<op>%in%|&&|\|\||!=|==|<=|>=|[&|!^])
    |(?P<other>\s+|.)
    """,
    re.VERBOSE | re.DOTALL,
)

_R_CONSTANTS = {
    "TRUE": "True",
    "T": "True",
    "FALSE": "False",
    "F": "False",
    "NULL": "None",
    "NA": "__NA__",
    "NA_real_": "__NA__",
    "NA_integer_": "__NA__",
    "NA_character_": "__NA__",
    "NaN": "__NA__",
    "Inf": "__Inf__",
}


def _not(x: Any) -> Any:
    if isinstance(x, (pd.Series, np.ndarray)):
        return ~x.astype(bool)
    return not x


def _isin(x: Any, values: Any) -> Any:
    values = list(values) if isinstance(values, (list, tuple, set, np.ndarray, pd.Series)) else [values]
    if isinstance(x, pd.Series):
        return x.isin(values)
    if isinstance(x, np.ndarray):
        return np.isin(x, values)
    return x in values


class _RToPython(ast.NodeTransformer):
    """Rewrite boolean keywords into element-wise operations so they work on Series."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        result = node.values[0]
        for value in node.values[1:]:
            result = ast.BinOp(left=result, op=op, right=value)
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.Call(func=ast.Name("__not__", ast.Load()), args=[node.operand], keywords=[])
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            call = ast.Call(func=ast.Name("__isin__", ast.Load()), args=[node.left, node.comparators[0]], keywords=[])
            if isinstance(node.ops[0], ast.NotIn):
                return ast.Call(func=ast.Name("__not__", ast.Load()), args=[call], keywords=[])
            return call
        return node


def _series(x: Any) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)


def _aggregate(method: str):
    def func(*args: Any, na_rm: bool = True, **kwargs: Any) -> Any:
        if len(args) == 1:
            return getattr(_series(args[0]), method)(skipna=na_rm)
        return getattr(pd.Series(np.concatenate([np.atleast_1d(a) for a in args])), method)(skipna=na_rm)

    return func


def _quantile(x: Any, probs: float = 0.5, na_rm: bool = True, **kwargs: Any) -> Any:
    if np.any(np.asarray(probs) > 1) or np.any(np.asarray(probs) < 0):
        raise ValueError("Quantile probabilities must be between 0 and 1")
    return _series(x).quantile(probs)


def _shift(x: Any, n: int = 1, default: Any = None) -> pd.Series:
    if default is None:
        return _series(x).shift(n)
    return _series(x).shift(n, fill_value=default)


def _lag(x: Any, n: int = 1, default: Any = None, **kwargs: Any) -> pd.Series:
    return _shift(x, n, default)


def _lead(x: Any, n: int = 1, default: Any = None, **kwargs: Any) -> pd.Series:
    return _shift(x, -n, default)


def _if_else(condition: Any, true: Any, false: Any, missing: Any = None, **kwargs: Any) -> Any:
    result = np.where(condition, true, false)
    if missing is not None:
        result = np.where(pd.isna(condition), missing, result)
    if isinstance(condition, pd.Series):
        return pd.Series(result, index=condition.index)
    return result


def _na_if(x: Any, y: Any) -> pd.Series:
    x = _series(x)
    return x.mask(x == y)


def _coalesce(*args: Any) -> Any:
    result = args[0]
    for value in args[1:]:
        if isinstance(result, pd.Series):
            result = result.fillna(value)
        elif pd.isna(result):
            result = value
    return result


def _log(x: Any, base: float = np.e) -> Any:
    return np.log(x) / np.log(base)


def _round(x: Any, digits: int = 0) -> Any:
    return np.around(x, digits)


def _mad(x: Any, constant: float = 1.4826, na_rm: bool = True, **kwargs: Any) -> float:
    x = _series(x).dropna() if na_rm else _series(x)
    return constant * (x - x.median()).abs().median()


def _nth(x: Any, n: int, default: Any = np.nan, **kwargs: Any) -> Any:
    x = _series(x)
    n = int(n)
    if n == 0 or n > len(x) or n < -len(x):
        return default
    return x.iloc[n - 1] if n > 0 else x.iloc[n]


_R_FUNCTIONS: dict = {
    # Aggregates
    "mean": _aggregate("mean"),
    "avg": _aggregate("mean"),
    "median": _aggregate("median"),
    "med": _aggregate("median"),
    "min": _aggregate("min"),
    "max": _aggregate("max"),
    "sum": _aggregate("sum"),
    "sd": _aggregate("std"),
    "var": _aggregate("var"),
    "quantile": _quantile,
    "IQR": lambda x, **kw: _quantile(x, 0.75) - _quantile(x, 0.25),
    "iqr": lambda x, **kw: _quantile(x, 0.75) - _quantile(x, 0.25),
    "mad": _mad,
    "n_distinct": lambda *args, **kw: pd.concat([_series(a) for a in args], axis=1).drop_duplicates().shape[0],
    "first": lambda x, **kw: _nth(x, 1, **kw),
    "last": lambda x, **kw: _nth(x, -1, **kw),
    "nth": _nth,
    # Element-wise maths
    "log": _log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "ceiling": np.ceil,
    "ceil": np.ceil,
    "floor": np.floor,
    "trunc": np.trunc,
    "round": _round,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    "pmin": lambda *args, **kw: np.minimum.reduce(args),
    "pmax": lambda *args, **kw: np.maximum.reduce(args),
    # Cumulative and window functions
    "cumsum": lambda x: _series(x).cumsum(),
    "cumprod": lambda x: _series(x).cumprod(),
    "cummean": lambda x: _series(x).expanding().mean(),
    "cummin": lambda x: _series(x).cummin(),
    "cummax": lambda x: _series(x).cummax(),
    "lag": _lag,
    "lead": _lead,
    # Logical and missing-value helpers
    "if_else": _if_else,
    "ifelse": _if_else,
    "is_na": pd.isna,
    "na_if": _na_if,
    "coalesce": _coalesce,
    "between": lambda x, left, right: _series(x).between(left, right),
    "recode": lambda x, *args, **kwargs: _series(x).replace(kwargs),
    "c": lambda *args: list(args),
}


def _translate_expression(expression: str, columns: list) -> tuple[str, dict, list]:
    """Translate an R expression into Python source, returning the source, column bindings and columns referenced."""
    column_index = {col: i for i, col in enumerate(columns)}
    bindings: dict = {}
    used: list = []
    tokens = [(m.lastgroup, m.group()) for m in _TOKEN_RE.finditer(expression)]
    pieces = []
    for i, (kind, text) in enumerate(tokens):
        following = "".join(t for _, t in tokens[i + 1 :]).lstrip()
        if kind == "string":
            pieces.append(text)
        elif kind == "number":
            pieces.append(text.rstrip("L"))
        elif kind in ("name", "backtick"):
            name = text.strip("`")
            is_call = following.startswith("(")
            is_keyword_arg = following.startswith("=") and not following.startswith("==")
            if kind == "name" and (is_call or is_keyword_arg):
                pieces.append(name.replace(".", "_"))
            elif name in column_index:
                placeholder = f"__col{column_index[name]}__"
                bindings[placeholder] = columns[column_index[name]]
                if name not in used:
                    used.append(name)
                pieces.append(placeholder)
            elif kind == "name" and name in _R_CONSTANTS:
                pieces.append(_R_CONSTANTS[name])
            elif kind == "backtick":
                raise KeyError(f"Column `{name}` not found in data")
            else:
                python_name = name.replace(".", "_")
                if keyword.iskeyword(python_name) and python_name not in ("True", "False", "None"):
                    python_name += "_"
                pieces.append(python_name)
        elif kind == "op":
            pieces.append(
                {"%in%": " in ", "&&": " and ", "&": " and ", "||": " or ", "|": " or ", "!": " not ", "^": "**"}.get(
                    text, text
                )
            )
        else:
            pieces.append(text)
    return "".join(pieces).strip(), bindings, used


def _check_safe(tree: ast.AST, expression: str, namespace: dict) -> None:
    """Reject Python constructs R expressions never need, so ``eval`` cannot reach object internals."""
    for node in ast.walk(tree):
        is_dunder = isinstance(node, ast.Name) and node.id.startswith("__") and node.id not in namespace
        if isinstance(node, ast.Attribute) or is_dunder:
            raise ValueError(f"Unsupported syntax in expression {expression!r}")


def _eval_expression(data: pd.DataFrame, expression: str) -> tuple[Any, list]:
    """Evaluate an R-style expression against the columns of ``data``.

    Returns the value of the expression and the names of the columns it referenced.
    """
    source, bindings, used = _translate_expression(expression, list(data.columns))
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Could not parse expression {expression!r}") from e
    tree = ast.fix_missing_locations(_RToPython().visit(tree))
    namespace = dict(_R_FUNCTIONS)
    namespace.update(
        {
            "__NA__": np.nan,
            "__Inf__": np.inf,
            "__not__": _not,
            "__isin__": _isin,
            "n": lambda: len(data),
            "row_number": lambda x=None: (
                pd.Series(np.arange(1, len(data) + 1), index=data.index)
                if x is None
                else _series(x).rank(method="first")
            ),
        }
    )
    namespace.update({placeholder: data[col] for placeholder, col in bindings.items()})
    _check_safe(tree, expression, namespace)
    try:
        value = eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}}, namespace)
    except NameError as e:
        raise ValueError(f"Unknown column or function in expression {expression!r}: {e}") from e
    return value, used


def _split_assignment(expression: str) -> tuple[str, str]:
    """Split ``"name = expression"`` into its name and expression."""
    match = re.match(r"^\s*(`[^`]+`|[^=<>!]+?)\s*=(?!=)(.*)$", expression, re.DOTALL)
    if match is None:
        raise ValueError(f"Expected an expression of the form 'name = value', got {expression!r}")
    return match.group(1).strip().strip("`"), match.group(2).strip()


def _as_mask(value: Any, index: pd.Index) -> pd.Series:
    """Convert the result of a filter condition into a boolean mask where missing values are False."""
    if np.ndim(value) == 0:
        return pd.Series(False if pd.isna(value) else bool(value), index=index)
    mask = value if isinstance(value, pd.Series) else pd.Series(np.asarray(value), index=index)
    return mask.fillna(False).astype(bool)


# One Table Verbs
def arrange(data: pd.DataFrame | ps.DataFrame, cols: str | list | tuple | np.ndarray) -> pd.DataFrame | ps.DataFrame:
    """Arrange rows by column values.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str or list
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
    columns = list(data.columns)
    if isinstance(cols, str):
        items = [cols]
    elif isinstance(cols, (list, tuple, np.ndarray)):
        items = list(cols)
    else:
        raise TypeError("Cannot determine method for determining column types")
    sorting_cols = []
    ascending_cols = []
    for item in items:
        desc_match = re.fullmatch(r"\s*desc\(\s*(.+?)\s*\)\s*", item)
        if item in columns:
            col, ascending = item, True
        elif desc_match is not None:
            col, ascending = desc_match.group(1).strip("`"), False
        elif item.startswith("-") and item[1:].strip() in columns:
            # dplyr also accepts arrange(-x) for a descending sort
            col, ascending = item[1:].strip(), False
        else:
            col, ascending = item, True
        if col not in columns:
            raise KeyError(f"Column {col!r} not found in data")
        sorting_cols.append(col)
        ascending_cols.append(ascending)
    if is_pandas:
        # R's implementation resets the index. kind="stable" keeps tied rows in their original order so the result is
        # deterministic across platforms and matches dplyr.
        return data.sort_values(sorting_cols, ascending=ascending_cols, kind="stable").reset_index(drop=True)
    else:
        return data.orderBy(sorting_cols, ascending=ascending_cols)


def count(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray = None,
    wt: str | list | tuple | np.ndarray = None,
    sort: bool = False,
    name: str = None,
    drop: bool = True,
) -> pd.Series:
    """Count observations by group.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str or list
         Optional variables to use when determining uniqueness. If there are multiple rows for a given combination of inputs,
         only the first row will be preserved. If None, will use all variables.
    wt : str or list, default is None
         Frequency weights. Can be a variable (or combination of variables) or None. wt is computed once for each unique
         combination of the counted variables.
    sort : bool, default is False
        If True, will show the largest groups at the top.
    name : str, default is None
        The name of the new column in the output. If omitted, it will default to n. If there's already a column called n,
        it will error, and require you to specify the name.
    drop : bool, default is True
        If False will include counts for empty groups (i.e. for levels of factors that don't exist in the data)

    Returns
    -------
    A pandas series with our counts or sums
    """
    is_pandas = _check_df_type(data, "count")
    distinct_cols = [] if cols is None else _resolve_columns(data, cols, is_pandas)
    if name is None:
        name = "n"
        while name in distinct_cols:
            name += "n"
    if is_pandas:
        if not distinct_cols:
            return pd.DataFrame({name: [len(data) if wt is None else data[wt].sum()]})
        # Grouping columns are treated as factors, as in R. Converting a copy keeps the caller's data untouched, and
        # existing categorical levels are preserved so that drop=False can report empty groups.
        grouping = data[distinct_cols].astype("category")
        if wt is not None:
            grouping = grouping.assign(__wt__=data[wt])
        grouped = grouping.groupby(distinct_cols, observed=drop, dropna=False)
        count_df = grouped.size() if wt is None else grouped["__wt__"].sum()
        count_df = count_df.rename(name)
        if sort:
            count_df = count_df.sort_values(ascending=False, kind="stable")
        count_df = count_df.reset_index()
    else:
        ...
    return count_df


def add_count(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray = None,
    wt: str | list | tuple | np.ndarray = None,
    sort: bool = False,
    name: str = None,
) -> pd.DataFrame | ps.DataFrame:
    """Count observations by group.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str or list
         Optional variables to use when determining uniqueness. If there are multiple rows for a given combination of inputs,
         only the first row will be preserved. If None, will use all variables.
    wt : str or list, default is None
         Frequency weights. Can be a variable (or combination of variables) or None. wt is computed once for each unique
         combination of the counted variables.
    sort : bool, default is False
        If True, will show the largest groups at the top.
    name : str, default is None
        The name of the new column in the output. If omitted, it will default to n. If there's already a column called n,
        it will error, and require you to specify the name.

    Returns
    -------
    Our dataframe, but with an additional column including the sum or count
    """
    is_pandas = _check_df_type(data, "add_count")
    distinct_cols = [] if cols is None else _resolve_columns(data, cols, is_pandas)
    if name is None:
        name = "n"
        while name in data.columns:
            name += "n"
    if is_pandas:
        data = data.copy()
        if not distinct_cols:
            data[name] = len(data) if wt is None else data[wt].sum()
        else:
            # Label each row with its group (keeping missing keys as their own group), then broadcast group totals
            group_ids = data.groupby(distinct_cols, dropna=False, sort=False).ngroup()
            if wt is None:
                data[name] = group_ids.map(group_ids.value_counts()).astype("int64")
            else:
                data[name] = data[wt].groupby(group_ids).transform("sum")
        if sort:
            data = data.sort_values(by=name, ascending=False, kind="stable")
    else:
        ...
    return data


def distinct(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray = None,
    keep_all: bool = False,
) -> pd.DataFrame | ps.DataFrame:
    """Subset distinct/unique rows.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str or list, default is None
         Optional variables to use when determining uniqueness. If there are multiple rows for a given combination of inputs,
         only the first row will be preserved. If None, will use all variables.
    keep_all : bool, default is False
        If True, keep all variables in data. If a combination of cols is not distinct, this keeps the first row of values.

    Returns
    -------
    Our dataframe but with unique rows specified
    """
    is_pandas = _check_df_type(data, "distinct")
    if cols is None:
        distinct_cols = list(data.columns) if is_pandas else data.schema.names
    else:
        distinct_cols = _resolve_columns(data, cols, is_pandas)
    if is_pandas:
        if keep_all:
            # Keep the first row of each distinct combination, with every column in its original position
            return data.drop_duplicates(subset=distinct_cols, keep="first", ignore_index=True)
        else:
            return data[distinct_cols].drop_duplicates(keep="first", ignore_index=True)
    else:
        ...


def filter(data: pd.DataFrame | ps.DataFrame, cols: str | list | tuple | np.ndarray) -> pd.DataFrame | ps.DataFrame:
    """Filter data based on arguments from cols.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        The dataframe for which we are filtering the data on
    cols : str or list
        The filter conditions we are applying on our dataframe

    Returns
    -------
    filtered_data : pandas DataFrame
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
    conditions = _as_list(cols)
    if is_pandas:
        # Each condition is evaluated on the full data (so aggregates like mean(x) see every row) and the conditions are
        # combined with AND. As in dplyr, rows where a condition evaluates to NA are dropped.
        mask = pd.Series(True, index=data.index)
        for condition in conditions:
            value, _ = _eval_expression(data, condition)
            mask &= _as_mask(value, data.index)
        return data[mask].reset_index(drop=True)
    else:
        return data.filter(" and ".join(f"({condition})" for condition in conditions))


def mutate(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray,
    keep: str = "all",
) -> pd.DataFrame | ps.DataFrame:
    """Add new variables while preserving existing ones.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str, list or dict
        Name-value pairs. The name gives the name of the column in the output. The value can be:
            A vector of length 1, which will be recycled to the correct length.
            A vector the same length as the current group (or the whole data frame if ungrouped).
            NULL, to remove the column.
            A data frame or tibble, to create multiple columns in the output.
    keep : str, default is all
        Allows you to control which columns from data are retained in the output:
            "all", the default, retains all variables.
            "used" keeps any variables used to make new variables; it's useful for checking your work as it displays inputs and outputs side-by-side.
            "unused" keeps only existing variables not used to make new variables.
            "none", only keeps grouping keys (like transmute()).

    Returns
    -------
    Our dataframe, but with the new columns added
    """
    is_pandas = _check_df_type(data, "mutate")
    keep = keep.casefold()
    if keep not in ("all", "used", "unused", "none"):
        raise ValueError("keep must be one of 'all', 'used', 'unused' or 'none'")
    if not is_pandas:
        raise NotImplementedError("mutate is not yet implemented for pyspark DataFrames")
    original_cols = list(data.columns)
    result = data.copy()
    used: set = set()
    created: list = []
    # Expressions are applied in order, so later expressions can refer to columns created by earlier ones
    for expression in _as_list(cols):
        name, rhs = _split_assignment(expression)
        value, referenced = _eval_expression(result, rhs)
        used.update(col for col in referenced if col in original_cols and col not in created)
        if value is None:
            # name = NULL removes the column
            result = result.drop(columns=name, errors="ignore")
            if name in created:
                created.remove(name)
            continue
        result[name] = value
        if name not in created:
            created.append(name)
    if keep == "all":
        return result
    if keep == "none":
        return result[[col for col in created if col in result.columns]]
    if keep == "used":
        return result[[col for col in result.columns if col in used or col in created]]
    return result[[col for col in result.columns if col not in used or col in created]]


def transmute(data, cols: str | list | tuple | np.ndarray | dict) -> pd.DataFrame | ps.DataFrame:
    """Add new variables while eliminating any columns not used to make our new columns.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str, list or dict
        Name-value pairs. The name gives the name of the column in the output. The value can be:
            A vector of length 1, which will be recycled to the correct length.
            A vector the same length as the current group (or the whole data frame if ungrouped).
            NULL, to remove the column.
            A data frame or tibble, to create multiple columns in the output.

    Returns
    -------
    Our dataframe, but with the new columns added and all non-grouping keys removed
    """
    return mutate(data, cols, keep="none")


def pull(
    data: pd.DataFrame | ps.DataFrame,
    var: int | str = -1,
    name: int | str = None,
) -> pd.Series | ps.Column:
    """Extract a single column.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    var : int or str, default is -1
        A variable specified as:
            * a literal variable name
            * a positive integer, giving the position counting from the left
            * a negative integer, giving the position counting from the right.
    name : int or str, default is None
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


def rename(data: pd.DataFrame | ps.DataFrame, cols: str | list | tuple | np.ndarray) -> pd.DataFrame | ps.DataFrame:
    """Change the name of individual columns.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str or list
        The columns we are renaming, and the name we are changing them to

    Returns
    -------
    Our dataframe, but with the columns renamed
    """
    is_pandas = _check_df_type(data, "rename")
    if is_pandas:
        rename_dict = {}
        for col in _as_list(cols):
            old_name, new_name = col.split("=", 1)
            rename_dict[old_name.strip().strip("`")] = new_name.strip().strip("`")
        data = data.rename(columns=rename_dict)
    else:
        ...
    return data


def relocate(
    data: pd.DataFrame | ps.DataFrame,
    cols: str | list | tuple | np.ndarray,
    before: str = None,
    after: str = None,
) -> pd.DataFrame | ps.DataFrame:
    """Use relocate() to change column positions.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str or list
         Columns to move
    before : str, default=None
        Destination of columns. Default is to move to the very left.
    after : str, default=None
        Destination of columns. Default is to move to the very left.

    Returns
    -------
    Our dataframe, but with the columns repositioned
    """
    is_pandas = _check_df_type(data, "relocate")
    cols = _resolve_columns(data, cols, is_pandas)
    if is_pandas:
        if before is not None and after is not None:
            raise ValueError("Only one of before or after can be specified")
        # cols is always a list here (see _get_str_columns / _get_list_columns).
        to_move = list(cols)
        remaining = [col for col in data.columns if col not in to_move]
        if before is None and after is None:
            # Default: move the columns to the very left, preserving original order.
            new_cols = to_move + remaining
        elif before is not None:
            insert_at = remaining.index(before)
            new_cols = remaining[:insert_at] + to_move + remaining[insert_at:]
        else:
            insert_at = remaining.index(after) + 1
            new_cols = remaining[:insert_at] + to_move + remaining[insert_at:]
        return data[new_cols]
    else:
        ...


def select(data: pd.DataFrame | ps.DataFrame, cols: str | list | tuple | np.ndarray) -> pd.DataFrame | ps.DataFrame:
    """Select variables in a data frame.

    Parameters
    ----------
    data : pandas or pyspark DataFrame
        A dataframe
    cols : str or list
         Columns to move

    Returns
    -------
    Our dataframe, but with the selected columns
    """
    is_pandas = _check_df_type(data, "select")
    cols = _resolve_columns(data, cols, is_pandas)
    if is_pandas:
        return data.loc[:, cols]
    else:
        return data.select(*cols)
