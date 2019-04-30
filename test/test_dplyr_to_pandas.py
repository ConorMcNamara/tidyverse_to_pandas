import unittest
import pandas as pd
import dplyr_to_pandas


class TestDplyrToPandas(unittest.TestCase):

    def test_mutate_nonDF_Exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, dplyr_to_pandas.mutate, data)

    def test_mutate_camelCase_result(self):
        df = pd.DataFrame(data={'aCol': [1, 3, 5, 7],
                                'bCol': [2, 4, 6, 8]})
        actual = dplyr_to_pandas.mutate(df, "cCol = aCol + bCol")
        expected = pd.DataFrame(data={'aCol': [1, 3, 5, 7],
                                      'bCol': [2, 4, 6, 8],
                                      'cCol': [3, 7, 11, 15]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_mutate_snakeCase_result(self):
        df = pd.DataFrame(data={'a_col': [1, 3, 5, 7],
                                'b_col': [2, 4, 6, 8]})
        actual = dplyr_to_pandas.mutate(df, "c_col = a_col * b_col")
        expected = pd.DataFrame(data={'a_col': [1, 3, 5, 7],
                                      'b_col': [2, 4, 6, 8],
                                      'c_col': [2, 12, 30, 56]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_mutate_camelCase1_result(self):
        df = pd.DataFrame(data={'col1': [1, 3, 5, 7],
                                'col2': [2, 4, 6, 8]})
        actual = dplyr_to_pandas.mutate(df, "col3 = col2 - col1")
        expected = pd.DataFrame(data={'col1': [1, 3, 5, 7],
                                      'col2': [2, 4, 6, 8],
                                      'col3': [1, 1, 1, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_transmute_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, dplyr_to_pandas.mutate, data)

    def test_transmute_camelCase_result(self):
        df = pd.DataFrame(data={'aCol': [1, 3, 5, 7],
                                'bCol': [2, 4, 6, 8]})
        actual = dplyr_to_pandas.transmute(df, "cCol = aCol + bCol")
        expected = pd.DataFrame(data={'cCol': [3, 7, 11, 15]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_transmute_snakeCase_result(self):
        df = pd.DataFrame(data={'a_col': [1, 3, 5, 7],
                                'b_col': [2, 4, 6, 8]})
        actual = dplyr_to_pandas.transmute(df, "c_col = a_col * b_col")
        expected = pd.DataFrame(data={'c_col': [2, 12, 30, 56]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_transmute_camelCase1_result(self):
        df = pd.DataFrame(data={'col1': [1, 3, 5, 7],
                                'col2': [2, 4, 6, 8]})
        actual = dplyr_to_pandas.transmute(df, "col3 = col2 - col1")
        expected = pd.DataFrame(data={'col3': [1, 1, 1, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_rename_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, dplyr_to_pandas.rename, data)

    def test_rename_DF_result(self):
        df = pd.DataFrame(data={'col1': [1, 2, 3 , 4]})
        actual = dplyr_to_pandas.rename(df, "col1 = col_1")
        expected = pd.DataFrame(data={'col_1': [1, 2, 3, 4]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_nonDF_exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, dplyr_to_pandas.select, data)

    def test_select_dropCol_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "-model")
        expected = pd.DataFrame({'mpg': [30, 50, 20, 25, 25],
                                 'vs': [1, 1, 0, 0, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_multipleCols_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "model", "mpg")
        expected = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                                 'mpg': [30, 50, 20, 25, 25]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_startsWith_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "starts_with(v)")
        expected = pd.DataFrame({'vs': [1, 1, 0, 0, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_endsWith_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "ends_with(g)")
        expected = pd.DataFrame({'mpg': [30, 50, 20, 25, 25]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_contains_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "contains(o)")
        expected = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW']})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_everything_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "everything()")
        pd.testing.assert_frame_equal(actual, df)

    def test_select_numRange_result(self):
        df = pd.DataFrame({'V1': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'V2': [30, 50, 20, 25, 25],
                           'V3': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "num_range(V, 2:3)")
        expected = pd.DataFrame({'V2': [30, 50, 20, 25, 25],
                                 'V3': [1, 1, 0, 0, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_lastColNoArgument_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "last_col()")
        expected = pd.DataFrame({'vs': [1, 1, 0, 0, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_lastColNumericArgument_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "last_col(1)")
        expected = pd.DataFrame({'mpg': [30, 50, 20, 25, 25]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_lastColOffsetSpaceArgument_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "last_col(offset = 1)")
        expected = pd.DataFrame({'mpg': [30, 50, 20, 25, 25]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_lastColOffsetArgument_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.select(df, "last_col(offset=2)")
        expected = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW']})
        pd.testing.assert_frame_equal(actual, expected)

    def test_select_tooManyOffsets_Exception(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        self.assertRaises(Exception, dplyr_to_pandas.select, df, "last_col(offset=3)")

    def test_filter_nonDF_Exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, dplyr_to_pandas.filter, data, "mpg > mean(mpg)")

    def test_filter_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.filter(df, "model == 'Lexus'")
        expected = pd.DataFrame({"model": ['Lexus'],
                                 "mpg": [25],
                                 "vs": [0]}, index=[3])
        pd.testing.assert_frame_equal(actual, expected)

    def test_filter_mean_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.filter(df, "mpg == mean(mpg)")
        expected = pd.DataFrame({"model": ['Mazda'],
                                 "mpg": [30],
                                 "vs": [1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_filter_median_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.filter(df, "mpg > median(mpg)")
        expected = pd.DataFrame({"model": ['Mazda', 'Toyota'],
                                 "mpg": [30, 50],
                                 "vs": [1, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_filter_min_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.filter(df, "mpg == min(mpg)")
        expected = pd.DataFrame({"model": ['Ford'],
                                 "mpg": [20],
                                 "vs": [0]}, index=[2])
        pd.testing.assert_frame_equal(actual, expected)

    def test_filter_max_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.filter(df, "mpg == max(vs)")
        expected = pd.DataFrame({"model": ['Mazda', 'Toyota', 'BMW'],
                                 "mpg": [30, 50, 25],
                                 "vs": [1, 1, 1]}, index=[0, 1, 4])
        pd.testing.assert_frame_equal(actual, expected)

    def test_filter_quantile_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.filter(df, "mpg  > quantile(mpg, 0.5)")
        expected = pd.DataFrame({"model": ['Mazda', 'Toyota'],
                                 "mpg": [30, 50],
                                 "vs": [1, 1]}, index=[0, 1])
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_nonDF_Exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, dplyr_to_pandas.summarise, data, "mean(mpg)")

    def test_summarise_mean_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "mean(mpg)")
        expected = pd.DataFrame(pd.Series(30.0, name='mean(mpg)'))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_median_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "median = median(mpg)")
        expected = pd.DataFrame(pd.Series(25.0, name='median'))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_sd_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "sd(mpg)")
        expected = pd.DataFrame(pd.Series(df['mpg'].std(), name="sd(mpg)"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_mad_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "mad = mad(mpg)")
        expected = pd.DataFrame(pd.Series(df['mpg'].mad(), name="mad"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_iqr_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "iqr(mpg)")
        expected = pd.DataFrame(pd.Series(5.0, name="iqr(mpg)"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_min_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "min(vs)")
        expected = pd.DataFrame(pd.Series(0, name="min(vs)"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_max_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "maximum = max(vs)")
        expected = pd.DataFrame(pd.Series(1, name="maximum"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_quantile_greaterThanOne_Exception(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        self.assertRaises(Exception, dplyr_to_pandas.summarise, df, "quantile(vs, 2")

    def test_summarise_quantile_fullDecimal_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "quant = quantile(vs, 0.5)")
        expected = pd.DataFrame(pd.Series(1.0, name="quant"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_quantile_partialDecimal_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "quant = quantile(vs, .5)")
        expected = pd.DataFrame(pd.Series(1.0, name="quant"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_first_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "first = first(model)")
        expected = pd.DataFrame(pd.Series("Mazda", name="first"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_last_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "last = last(model)")
        expected = pd.DataFrame(pd.Series("BMW", name="last"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_nth_PositiveOutsideBounds_Exception(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        self.assertRaises(Exception, dplyr_to_pandas.summarise, df, "nth(mpg, 6)")

    def test_summarise_nth_NegativeOutsideBounds_Exception(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        self.assertRaises(Exception, dplyr_to_pandas.summarise, df, "nth(mpg, -6)")

    def test_summarise_nth_positiveInt_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "nth(model, 4)")
        expected = pd.DataFrame(pd.Series("Lexus", name="nth(model, 4)"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_nth_negativeInt_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "nth(model, -4)")
        expected = pd.DataFrame(pd.Series("Toyota", name="nth(model, -4)"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_n_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "n()")
        expected = pd.DataFrame(pd.Series(5, name="n()"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_summarise_nDistinct_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.summarise(df, "distinct = n_distinct(mpg)")
        expected = pd.DataFrame(pd.Series(4, name="distinct"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_arrange_nonDF_Exception(self):
        data = [1, 2, 3, 4]
        self.assertRaises(Exception, dplyr_to_pandas.arrange, data, "portal")

    def test_arrange_noDesc_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.arrange(df, "model")
        expected = pd.DataFrame({'model': ['BMW', 'Ford', 'Lexus', 'Mazda', 'Toyota'],
                                 'mpg': [25, 20, 25, 30, 50],
                                 'vs': [1, 0, 0, 1, 1]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_arrange_desc_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.arrange(df, "desc(mpg)")
        expected = pd.DataFrame({"model": ["Toyota", "Mazda", "Lexus", "BMW", "Ford"],
                                 "mpg": [50, 30, 25, 25, 20],
                                 "vs": [1, 1, 0, 1, 0]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_arrange_both_result(self):
        df = pd.DataFrame({'model': ['Mazda', 'Toyota', 'Ford', 'Lexus', 'BMW'],
                           'mpg': [30, 50, 20, 25, 25],
                           'vs': [1, 1, 0, 0, 1]})
        actual = dplyr_to_pandas.arrange(df, "mpg", "desc(vs)")
        expected = pd.DataFrame({"model": ['Ford', 'BMW', 'Lexus', 'Mazda', 'Toyota'],
                                 'mpg': [20, 25, 25, 30, 50],
                                 'vs': [0, 1, 0, 1, 1]})
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
