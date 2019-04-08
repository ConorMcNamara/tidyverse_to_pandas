# tidyverse_to_pandas
Attempts to provide tidyverse syntax (such as gather() and spread()) using pandas DataFrames, however, this does not contain support for magrittr's piping functionality.

Instead, after importing either tidyr_to_pandas or dplyr_to_pandas, you call the function and provide arguments as strings.

For example, given a dataframe df, I would then call 
`` transmute(df, 'col_C = col_A + col_B')``


## Basics
There are currently two main source files for tidyverse_to_pandas: tidyr_to_pandas and dplyr_to_pandas.

### tidyr_to_pandas
Contains support for spread(), gather(), separate(), unite() and extract() from tidyr.

### dplyr_to_pandas
Contains support for mutate(), transmute(), rename(), select(), filter(), summarise() and arrange(). 
