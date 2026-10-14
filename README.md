# tidyverse-to-pandas

[![Tests](https://github.com/ConorMcNamara/tidyverse_to_pandas/actions/workflows/tests.yml/badge.svg)](https://github.com/ConorMcNamara/tidyverse_to_pandas/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)
![License](https://img.shields.io/badge/license-GPL--3.0--or--later-green)

The tidyverse's verbs for data manipulation, written for pandas.

pandas is powerful, but turning "keep the 4-cylinder cars and sort them by mpg" into code often means a trip to the
docs. This package provides functions named after the tidyverse's `dplyr`, `tidyr`, `stringr` and `lubridate`, with
argument names to match, so the way you think about the data in R carries over to Python.

There is no pipe operator (`%>%`). Each function takes a DataFrame (or a string, list, array or Series) as its first
argument and returns a new result.

## Installation

Requires Python 3.13+ and pandas 3.

```bash
pip install tidyverse-to-pandas
# or
uv add tidyverse-to-pandas
```

## Quick start

```python
import pandas as pd

from tidyverse_to_pandas import lubridate_to_pandas as lubridate
from tidyverse_to_pandas import rebase_dplyr_to_pandas as dplyr
from tidyverse_to_pandas import stringr_to_pandas as stringr
from tidyverse_to_pandas import tidyr_to_pandas as tidyr

mtcars = pd.read_csv("data/mtcars.csv")

# dplyr: filter, add a column, sort, then select
four_cyl = dplyr.filter(mtcars, "cyl == 4 & mpg > mean(mpg)")
four_cyl = dplyr.mutate(four_cyl, "wt_lbs = wt * 1000")
four_cyl = dplyr.arrange(four_cyl, ["desc(mpg)"])
dplyr.select(four_cyl, ["mpg", "cyl", "wt_lbs"])

dplyr.count(mtcars, "cyl")

# tidyr: wide -> long
religion = pd.read_csv("data/religion.csv")
tidyr.pivot_longer(religion, "-religion", names_to="income", values_to="count")

# stringr
stringr.str_detect(pd.Series(["apple", "banana"]), "an")  # [False, True]
stringr.str_pad("7", 3, side="left", pad="0")  # "007"

# lubridate
lubridate.mdy("January 5, 2024")  # "2024-01-05"
lubridate.dmy(pd.Series(["05/01/2024"]))  # Series of datetime64
```

### Column selection

Wherever a function takes column names, you can use tidyselect-style strings:

| Syntax    | Meaning                           |
|-----------|-----------------------------------|
| `"x"`     | the column `x`                    |
| `"-x"`    | every column except `x`           |
| `"x:y"`   | every column from `x` through `y` |

`helper_select` also provides `everything(df)` and `last_col(df, offset=0)`.

## Supported functions

The functions below are implemented. Each name links to its R documentation, but behavior can differ from R in places
(see [Limitations](#limitations)).

### dplyr — `tidyverse_to_pandas.rebase_dplyr_to_pandas`

| Function | Description |
|---|---|
| [`arrange`](https://dplyr.tidyverse.org/reference/arrange.html) | Order rows by column values; wrap a column in `desc()` to sort descending |
| [`count`](https://dplyr.tidyverse.org/reference/count.html) | Count observations by group |
| [`add_count`](https://dplyr.tidyverse.org/reference/count.html) | Add the group count as a new column |
| [`distinct`](https://dplyr.tidyverse.org/reference/distinct.html) | Keep distinct/unique rows |
| [`filter`](https://dplyr.tidyverse.org/reference/filter.html) | Keep rows that match a condition, e.g. `"mpg > mean(mpg)"` |
| [`mutate`](https://dplyr.tidyverse.org/reference/mutate.html) | Add, modify or remove (`"x = NULL"`) columns, e.g. `"ratio = hp / wt"` |
| [`transmute`](https://dplyr.tidyverse.org/reference/transmute.html) | Add new columns and drop the rest |
| [`pull`](https://dplyr.tidyverse.org/reference/pull.html) | Extract a single column as a Series |
| [`rename`](https://dplyr.tidyverse.org/reference/rename.html) | Rename columns, e.g. `"old = new"` (note: the reverse of dplyr's `new = old`) |
| [`relocate`](https://dplyr.tidyverse.org/reference/relocate.html) | Change column order |
| [`select`](https://dplyr.tidyverse.org/reference/select.html) | Keep or drop columns |

`filter`, `mutate` and `transmute` take R expressions as strings. They support R operators (`& | ! %in% ^`),
`TRUE`/`FALSE`/`NA`, dotted or backtick-quoted column names, and common R functions such as `mean`, `median`, `sd`,
`quantile`, `n()`, `row_number()`, `lag`/`lead`, `if_else`, `coalesce`, `is.na` and `between`. Passing a list
applies the expressions in order (for `filter`, the conditions are combined with AND).

`tidyverse_to_pandas.dplyr_to_pandas` is the older `(data, *args)` version of this API. It also has `summarise` and
the `starts_with` / `ends_with` / `contains` / `num_range` selection helpers. Use `rebase_dplyr_to_pandas` for new
code.

### tidyr — `tidyverse_to_pandas.tidyr_to_pandas`

| Category | Functions |
|---|---|
| Pivoting | [`pivot_longer`](https://tidyr.tidyverse.org/reference/pivot_longer.html), [`pivot_wider`](https://tidyr.tidyverse.org/reference/pivot_wider.html) |
| Rectangling | [`unnest_longer`](https://tidyr.tidyverse.org/reference/unnest_longer.html), [`unnest_wider`](https://tidyr.tidyverse.org/reference/unnest_wider.html) |
| Nesting | [`nest`](https://tidyr.tidyverse.org/reference/nest.html), [`unnest`](https://tidyr.tidyverse.org/reference/unnest.html) |
| Chopping | [`chop`](https://tidyr.tidyverse.org/reference/chop.html), [`unchop`](https://tidyr.tidyverse.org/reference/chop.html) |
| Character columns | [`separate`](https://tidyr.tidyverse.org/reference/separate.html), [`extract`](https://tidyr.tidyverse.org/reference/extract.html), [`unite`](https://tidyr.tidyverse.org/reference/unite.html) |
| Missing values | [`drop_na`](https://tidyr.tidyverse.org/reference/drop_na.html), [`replace_na`](https://tidyr.tidyverse.org/reference/replace_na.html), [`fill`](https://tidyr.tidyverse.org/reference/fill.html), [`complete`](https://tidyr.tidyverse.org/reference/complete.html) |

### stringr — `tidyverse_to_pandas.stringr_to_pandas`

Most functions accept a `str`, a list/tuple, a NumPy array or a pandas Series.

| Category | Functions |
|---|---|
| Characters | [`str_length`](https://stringr.tidyverse.org/reference/str_length.html), [`str_sub`](https://stringr.tidyverse.org/reference/str_sub.html), [`str_dup`](https://stringr.tidyverse.org/reference/str_dup.html), [`str_c`](https://stringr.tidyverse.org/reference/str_c.html), [`str_flatten`](https://stringr.tidyverse.org/reference/str_flatten.html), [`str_trunc`](https://stringr.tidyverse.org/reference/str_trunc.html), [`str_replace_na`](https://stringr.tidyverse.org/reference/str_replace_na.html), [`str_unique`](https://stringr.tidyverse.org/reference/str_unique.html) |
| Case | [`str_to_upper`, `str_to_lower`, `str_to_title`, `str_to_sentence`](https://stringr.tidyverse.org/reference/case.html) |
| Order & equality | [`str_order`, `str_sort`](https://stringr.tidyverse.org/reference/str_order.html), [`str_equal`](https://stringr.tidyverse.org/reference/str_equal.html) |
| Whitespace | [`str_pad`](https://stringr.tidyverse.org/reference/str_pad.html), [`str_trim`, `str_squish`](https://stringr.tidyverse.org/reference/str_trim.html) |
| Detect | [`str_detect`](https://stringr.tidyverse.org/reference/str_detect.html), [`str_count`](https://stringr.tidyverse.org/reference/str_count.html), [`str_subset`, `str_which`](https://stringr.tidyverse.org/reference/str_subset.html), [`str_starts`, `str_ends`](https://stringr.tidyverse.org/reference/str_starts.html) |
| Replace & remove | [`str_replace`, `str_replace_all`](https://stringr.tidyverse.org/reference/str_replace.html), [`str_remove`, `str_remove_all`](https://stringr.tidyverse.org/reference/str_remove.html) |
| Split | [`str_split`, `str_split_fixed`, `str_split_n`](https://stringr.tidyverse.org/reference/str_split.html) |
| Extract & match | [`str_extract`, `str_extract_all`](https://stringr.tidyverse.org/reference/str_extract.html), [`str_match`, `str_match_all`](https://stringr.tidyverse.org/reference/str_match.html) |

### lubridate — `tidyverse_to_pandas.lubridate_to_pandas`

[`ymd`, `ydm`, `mdy`, `myd`, `dmy`, `dym`](https://lubridate.tidyverse.org/reference/ymd.html) parse dates in the
named field order, and [`ymd_hms`](https://lubridate.tidyverse.org/reference/ymd_hms.html) parses date-times. They
accept a `str`, list/tuple, NumPy array or Series, plus an optional `tz`:

- Any separator works, as do month names (`"5 Sept 2024"`) and compact digits (`"20240905"`).
- Two-digit years follow lubridate: `00`–`68` become 20xx and `69`–`99` become 19xx.
- Strings and lists come back as `"YYYY-MM-DD"` strings, arrays as `datetime64`, and Series as `datetime64[ns]`
  (tz-aware if `tz` is set).
- Values that don't fit the requested order (e.g. `mdy("13/01/2020")`) come back missing, with a warning.

## Limitations

- **pandas only.** The functions have pyspark branches and a `spark` extra (`pip install tidyverse-to-pandas[spark]`),
  but the Spark code paths are not working yet. Pass pandas objects.
- **Expressions are strings.** `filter`, `mutate` and `transmute` translate R expressions to pandas. Only the R
  functions listed above are available, and an unknown function or column raises `ValueError`.
- **Not a full port.** Behavior follows the R documentation where it can, but some arguments are not implemented and
  some edge cases (missing values, non-default indexes) differ from R. Please
  [open an issue](https://github.com/ConorMcNamara/tidyverse_to_pandas/issues) when you find one.

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install the package and dev dependencies
uv run pytest           # run the test suite
uv run ruff check .     # lint
uv run ruff format .    # format
```

CI runs the tests on every Python version listed in `.github/python-versions.json`.

## License

[GPL-3.0-or-later](LICENSE)
