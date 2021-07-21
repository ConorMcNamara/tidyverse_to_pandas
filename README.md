# tidyverse_to_pandas
The goal here is simple: the tidyverse provides a nice opinionated way of manipulating data. And while pandas and pyspark do a good 
job for the most part, the area that they struggle in is simplicity. These packages are very powerful, but they lack the ability
to easily translate thought processes into actionable code quickly.

This package is designed to address those concerns by mimicking tidyverse functions to provide ease of use. While it does
not have any kind of support for magrittr's pipe function, it still supports using the main functions of tidyverse as is.
Thus, the user gets to spend more time writing scripts for feature engineering, data manipulation and exploratory data analysis
than they do scouring StackOverflow or pandas/pyspark documentation, which is a win for everyone.

## tidyr
To quote Hadley Wickham, "The goal of tidyr is to help you create tidy data. Tidy data is data where:

1. Every column is variable.
2. Every row is an observation..
3. Every cell is a single value.

Tidy data describes a standard way of storing data that is used wherever possible throughout the tidyverse. If you ensure that your data is tidy, you’ll spend less time fighting with the tools and more time working on your analysis."

Currently, I have support for the following tidyr functions:

### Pivoting Data
* [pivot_longer](https://tidyr.tidyverse.org/reference/pivot_longer.html): Converts wide data into long data
* [pivot_wider](https://tidyr.tidyverse.org/reference/pivot_wider.html): Converts long data into wide data

### Rectangling Data
* [unnest_longer](https://tidyr.tidyverse.org/reference/hoist.html): Converts deeply nested jsons into long data
* [unnest_wider](https://tidyr.tidyverse.org/reference/hoist.html): Converts deeply nested jsons into wide data

### Chopping Data
* [chop](https://tidyr.tidyverse.org/reference/chop.html): Converts rows within each group into a list of lists
* [unchop](https://tidyr.tidyverse.org/reference/chop.html): Expands a list of lists into new rows and columns

### Nesting
* [nest](https://tidyr.tidyverse.org/reference/nest.html): Converts each row into a nested dictionary/JSON
* [unnest](https://tidyr.tidyverse.org/reference/nest.html): Expands each nested row into new rows and columns

### Splitting and Combining Character Columns
* [separate](https://tidyr.tidyverse.org/reference/separate.html): Pulls a single character column into multiple columns using a known separator
* [extract](https://tidyr.tidyverse.org/reference/extract.html): Pulls a single character column into multiple columns using regular expressions
* [unite](https://tidyr.tidyverse.org/reference/unite.html): Combine multiple columns into a single character column

### Handling Missing Data
* [drop_na](https://tidyr.tidyverse.org/reference/drop_na.html): Makes explicit missing values implicit
* [fill](https://tidyr.tidyverse.org/reference/fill.html): Replace missing values with next/previous value
* [replace_na](https://tidyr.tidyverse.org/reference/replace_na.html): Replace missing values with a known value
* [complete](https://tidyr.tidyverse.org/reference/complete.html): Turns implicit missing values into explicit missing values

## dplyr
To quote Hadley Wickham, "dplyr is a grammar of data manipulation, providing a consistent set of verbs that help you solve the most common data manipulation challenges:"

Currently I have support for the following dplyr functions:

### One Table Verbs
* [arrange](https://dplyr.tidyverse.org/reference/arrange.html): Arrange rows by column values
* [count](https://dplyr.tidyverse.org/reference/count.html): Count observations by group
* [add_count](https://dplyr.tidyverse.org/reference/count.html): Count observations by group into new column
* [distinct](https://dplyr.tidyverse.org/reference/distinct.html): Subset distinct/unique rows
* [filter](https://dplyr.tidyverse.org/reference/filter.html): Subset rows using column values
* [mutate](https://dplyr.tidyverse.org/reference/mutate.html): Adds new variables and preserves existing ones
* [transute](https://dplyr.tidyverse.org/reference/mutate.html): Adds new variables and drops existing ones
* [pull](https://dplyr.tidyverse.org/reference/pull.html): Extract a single column
* [rename](https://dplyr.tidyverse.org/reference/rename.html): Rename a column(s)
* [relocate](https://dplyr.tidyverse.org/reference/relocate.html): Change column order


## stringr
To quote Hadley Wickham, "Strings are not glamorous, high-profile components of R, but they do play a big role in many data cleaning and preparation tasks. 
The stringr package provide a cohesive set of functions designed to make working with strings as easy as possible. If you’re not familiar with strings, 
the best place to start is the chapter on strings in R for Data Science."

### Character Manipulation
* [str_length](https://stringr.tidyverse.org/reference/str_length.html): Returns the number of characters in a string
* [str_sub](https://stringr.tidyverse.org/reference/str_sub.html): Extracts substrings from a string
* [str_dup](https://stringr.tidyverse.org/reference/str_dup.html): Duplicates a string
* [str_flatten](https://stringr.tidyverse.org/reference/str_flatten.html): Flattens a string
* [str_trunc](https://stringr.tidyverse.org/reference/str_trunc.html): Truncates a string
* [str_replace_na](https://stringr.tidyverse.org/reference/str_replace_na.html): Replaces NaN/None with a string

### Case Transformation
* [str_to_upper](https://stringr.tidyverse.org/reference/case.html): Converts the entire string to uppercase
* [str_to_lower](https://stringr.tidyverse.org/reference/case.html): Converts the entire string to lowercase
* [str_to_title](https://stringr.tidyverse.org/reference/case.html): Converts the entire string to title format
* [str_to_sentence](https://stringr.tidyverse.org/reference/case.html): Converts the entire string to sentence format

### Order
* [str_order](https://stringr.tidyverse.org/reference/str_order.html): Orders a string
* [str_sort](https://stringr.tidyverse.org/reference/str_order.html): Sorts a string

### Whitespace Manipulation
* [str_pad](https://stringr.tidyverse.org/reference/str_pad.html): Pads a string
* [str_trim](https://stringr.tidyverse.org/reference/str_trim.html): Removes whitespace from start and end of a string
* [str_squish](https://stringr.tidyverse.org/reference/str_trim.html): In addition to trimming, also reduces repeated whitespace inside a string

### Pattern Matching
* [str_detect](https://stringr.tidyverse.org/reference/str_detect.html): Determines if a string contains the regular expression
* [str_count](https://stringr.tidyverse.org/reference/str_count.html): Counts the number of matches in a string
* [str_subset](https://stringr.tidyverse.org/reference/str_subset.html): Keeps strings matching a pattern
* [str_which](https://stringr.tidyverse.org/reference/str_subset.html): Finds positions matching a pattern
* [str_replace](https://stringr.tidyverse.org/reference/str_replace.html): Replace the first matched pattern in a string
* [str_replace_all](https://stringr.tidyverse.org/reference/str_replace.html): Replace all matched patterns in a string
* [str_remove](https://stringr.tidyverse.org/reference/str_remove.html): Removes the first matched pattern in a string
* [str_remove_all](https://stringr.tidyverse.org/reference/str_remove.html): Removes all matched patterns in a string
* [str_split](https://stringr.tidyverse.org/reference/str_split.html): Splits a string into multiple pieces
* [str_split_fixed](https://stringr.tidyverse.org/reference/str_split.html): Splits a string into a character matrix
* [str_split_n](https://stringr.tidyverse.org/reference/str_split.html): Splits a string into a character vector
* [str_starts](https://stringr.tidyverse.org/reference/str_starts.html): Detect the presence of a pattern at the beginning of a string
* [str_ends](https://stringr.tidyverse.org/reference/str_starts.html): Detect the presence of a pattern at the end of a string
* [str_extract](https://stringr.tidyverse.org/reference/str_extract.html): Extract first matching pattern from a string
* [str_extract_all](https://stringr.tidyverse.org/reference/str_extract.html): Extract all non-overlapping matching patterns from a string
* [str_match](https://stringr.tidyverse.org/reference/str_match.html): Extracts first matched group in a string
* [str_match_all](https://stringr.tidyverse.org/reference/str_match.html): Extracts all matched groups in a string


## lubridate
To quote Hadley Wickham, "Date-time data can be frustrating to work with in R. R commands for date-times are generally unintuitive and change depending on the type of date-time object being used. Moreover, the methods we use with date-times must be robust to time zones, leap days, daylight savings times, and other time related quirks, and R lacks these capabilities in some situations. Lubridate makes it easier to do the things R does with date-times and possible to do the things R does not."

Currently I have support for the following lubridate functions:
* [ymd](https://lubridate.tidyverse.org/reference/ymd.html): Transforms dates in the format ymd
* [ydm](https://lubridate.tidyverse.org/reference/ymd.html): Transforms dates in the format ydm
* [mdy](https://lubridate.tidyverse.org/reference/ymd.html): Transforms dates in the format mdy
* [myd](https://lubridate.tidyverse.org/reference/ymd.html): Transforms dates in the format myd
* [dmy](https://lubridate.tidyverse.org/reference/ymd.html): Transforms dates in the format dmy
* [dym](https://lubridate.tidyverse.org/reference/ymd.html): Transforms dates in the format dym
