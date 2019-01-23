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
