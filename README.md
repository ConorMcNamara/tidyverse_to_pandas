# tidyverse_to_pandas
Currently going through an entire re-write/re-base. 

Essentially, back-end was a hodge-podge of spaghetti code that "worked" but not well. Additionally, recent changes to tidyr meant that creating new functions 
to mimic functionality was more difficult than it needed to be, due to lack of common software principles such as Single Responsibility. 

So I'm currently re-writing the entire back-end to better mimic tidyr syntax, as well as make it much easier to create new functions, and include support for pyspark as well as pandas DataFrames.

## tidyr
To quote Hadley Wickham, "The goal of tidyr is to help you create tidy data. Tidy data is data where:

1. Every column is variable.
2. Every row is an observation..
3. Every cell is a single value.

Tidy data describes a standard way of storing data that is used wherever possible throughout the tidyverse. If you ensure that your data is tidy, you’ll spend less time fighting with the tools and more time working on your analysis."

Currently, I have support for the following tidyr functions:

### Handling Missing Data
* [drop_na](https://tidyr.tidyverse.org/reference/drop_na.html): Makes explicit missing values implicit
* [fill](https://tidyr.tidyverse.org/reference/fill.html): Replace missing values with next/previous value
* [replace_na](https://tidyr.tidyverse.org/reference/replace_na.html): Replace missing values with a knonw value

### Splitting and Combining Character Columns
* [separate](https://tidyr.tidyverse.org/reference/separate.html): Pulls a single character column into multiple columns using a known separator
* [extract](https://tidyr.tidyverse.org/reference/extract.html): Pulls a single character column into multiple columns using regular expressions
* [unite](https://tidyr.tidyverse.org/reference/unite.html): Combine multiple columns into a single character column

### Pivoting Data
* [pivot_longer](https://tidyr.tidyverse.org/reference/pivot_longer.html): Converts wide data into long data
* [pivot_wider](https://tidyr.tidyverse.org/reference/pivot_wider.html): Converts long data into wide data

## dplyr
To quote Hadley Wickham, "dplyr is a grammar of data manipulation, providing a consistent set of verbs that help you solve the most common data manipulation challenges:"
* [mutate](https://dplyr.tidyverse.org/reference/mutate.html): Adds new variables that are functions of existing variables
* [select](https://dplyr.tidyverse.org/reference/select.html): Picks variables based on their names
* [filter](https://dplyr.tidyverse.org/reference/filter.html): Picks cases based on their values
* [summarise](https://dplyr.tidyverse.org/reference/summarise.html): Reduces multiple values down to a single summary
* [arrange](https://dplyr.tidyverse.org/reference/arrange.html): Changes the ordering of the rows