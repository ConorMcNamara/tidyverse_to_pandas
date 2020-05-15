import unittest
import pytest
import pandas as pd
import numpy as np
from src.stringr_to_pandas import str_length, str_sub, str_detect, str_count, str_dup, str_subset, str_to_upper, \
    str_to_lower, str_to_sentence, str_to_title, str_replace, str_order, str_sort, str_flatten, str_trunc


class TestStringrToPandas(unittest.TestCase):

    # String Length
    def test_strLength_string(self):
        string = 'abcd'
        assert str_length(string) == 4

    def test_strLength_list(self):
        string = ['ab', 'cdef']
        assert str_length(string) == [2, 4]

    def test_strLength_array(self):
        string = np.array(['abc', 'defg'])
        np.testing.assert_array_equal(str_length(string), np.array([3, 4]))

    def test_strLength_series(self):
        string = pd.Series(['abcd', 'efg'])
        pd.testing.assert_series_equal(str_length(string), pd.Series([4, 3]))

    # String Sub
    def test_strSub_string(self):
        string = 'abcd'
        assert str_sub(string, 0, 3) == 'abc'

    def test_strSub_list(self):
        string = ["abcdef", "ghifjk"]
        assert str_sub(string, 2, 3) == ['c', 'i']

    def test_strSub_array(self):
        string = np.array(["abcdef", "ghifjk"])
        np.testing.assert_array_equal(str_sub(string, 1, -1), np.array(['bcde', 'hifj']))

    def test_strSub_series(self):
        string = pd.Series(["abcdef", "ghifjk"])
        pd.testing.assert_series_equal(str_sub(string, 2, -1), pd.Series(['cde', 'ifj']))

    # String Duplicate
    def test_strDup_string(self):
        string = 'abcd'
        assert str_dup(string, 2) == 'abcdabcd'

    def test_strDup_list(self):
        string = ['aa', 'bb']
        assert str_dup(string, 2) == ['aaaa', 'bbbb']

    def test_strDup_array(self):
        string = np.array(['zz', 'yy'])
        np.testing.assert_array_equal(str_dup(string, [2, 3]), np.array(['zzzz', 'yyyyyy']))

    def test_strDup_series(self):
        string = pd.Series(['abc', 'xyz'])
        pd.testing.assert_series_equal(str_dup(string, 3), pd.Series(['abcabcabc', 'xyzxyzxyz']))

    # String Flatten
    def test_strFlatten_string(self):
        string = 'Mass Effect: Andromeda'
        assert str_flatten(string, ",") == string

    def test_strFlatten_list(self):
        string = ['This', 'is', 'a', 'test']
        assert str_flatten(string, " ") == 'This is a test'

    def test_strFlatten_array(self):
        string = np.array(["The", 'only', 'thing', 'they', 'fear', 'is', 'you'])
        assert str_flatten(string, ".") == "The.only.thing.they.fear.is.you"

    def test_strFlatten_series(self):
        string = pd.Series(["A", "B", "B", "A"])
        assert str_flatten(string) == "ABBA"

    # String Truncate
    def test_strTrunc_string(self):
        string = "Mass Effect: Andromeda"
        assert str_trunc(string, 11, "right") == "Mass Effect..."

    def test_strTrunc_list(self):
        string = ["This  string  is  moderately  long", "Guinea  pigs  and  farts"]
        assert str_trunc(string, 20, "center") == ['This  stri...tely  long', 'Guinea  pi...and  farts']

    def test_strTrunc_array(self):
        string = np.array(["This  string  is  moderately  long", "Guinea  pigs  and  farts"])
        expected = np.array(['...is  moderately  long', '...ea  pigs  and  farts'])
        np.testing.assert_array_equal(str_trunc(string, 20, "left"), expected)

    def test_strTrunc_series(self):
        string = pd.Series(["This  string  is  moderately  long", "Guinea  pigs  and  farts"])
        expected = pd.Series(["This  string  is  mo...", "Guinea  pigs  and  f..."])
        pd.testing.assert_series_equal(str_trunc(string, 20), expected)

    # String Uppercase
    def test_strToUpper_string(self):
        string = 'aabb'
        assert str_to_upper(string) == 'AABB'

    def test_strToUpper_list(self):
        string = ['a', 'ab', 'abc']
        assert str_to_upper(string) == ['A', 'AB', 'ABC']

    def test_strToUpper_array(self):
        string = np.array(['video', 'killed', 'radio', 'star'])
        np.testing.assert_array_equal(str_to_upper(string), np.array(['VIDEO', 'KILLED', 'RADIO', 'STAR']))

    def test_strToUpper_series(self):
        string = pd.Series(['green', 'lantern', 'first', 'flight'])
        pd.testing.assert_series_equal(str_to_upper(string), pd.Series(['GREEN', 'LANTERN', 'FIRST', 'FLIGHT']))

    # String Lowercase
    def test_strToLower_string(self):
        string = 'ABBA'
        assert str_to_lower(string) == 'abba'

    def test_strToLower_list(self):
        string = ['DANCING', 'QUEEN']
        assert str_to_lower(string) == ['dancing', 'queen']

    def test_strToLower_array(self):
        string = np.array(['I', 'AM', 'THE', 'LAW'])
        np.testing.assert_array_equal(str_to_lower(string), np.array(['i', 'am', 'the', 'law']))

    def test_strToLower_series(self):
        string = pd.Series(['WHY', 'IS', 'THIS', 'HAPPENING'])
        pd.testing.assert_series_equal(str_to_lower(string), pd.Series(['why', 'is', 'this', 'happening']))

    # String To Title
    def test_strToTitle_string(self):
        string = 'what is going on here?'
        assert str_to_title(string) == 'What Is Going On Here?'

    def test_strToTitle_list(self):
        string = ['i hate this', 'so much']
        assert str_to_title(string) == ['I Hate This', 'So Much']

    def test_strToTitle_array(self):
        string = np.array(['the', 'flight', 'of', 'fear'])
        np.testing.assert_array_equal(str_to_title(string), np.array(['The', 'Flight', 'Of', 'Fear']))

    def test_strToTitle_series(self):
        string = pd.Series(['guinea pigs', 'and farts'])
        pd.testing.assert_series_equal(str_to_title(string), pd.Series(['Guinea Pigs', 'And Farts']))

    # String To Sentence
    def test_strToSentence_string(self):
        string = 'this is a sentence'
        assert str_to_sentence(string) == 'This is a sentence'

    def test_strToSentence_list(self):
        string = ['michael eisner', 'roy disney']
        assert str_to_sentence(string) == ['Michael eisner', 'Roy disney']

    def test_strToSentence_array(self):
        string = np.array(['what is love', "baby don't hurt me"])
        np.testing.assert_array_equal(str_to_sentence(string), np.array(['What is love', "Baby don't hurt me"]))

    def test_strToSentence_series(self):
        string = pd.Series(["that's the sound your mother made", "lasht night"])
        pd.testing.assert_series_equal(str_to_sentence(string), pd.Series(["That's the sound your mother made", "Lasht night"]))

    # String Order
    def test_strOrder_string(self):
        string = 'string'
        assert str_order(string) == 0

    def test_strOrder_list(self):
        string = ['i', 'am', 'a', 'string']
        assert str_order(string) == [2, 1, 0, 3]

    def test_strOrder_array(self):
        string = np.array(['this', 'is', 'happening'])
        np.testing.assert_array_equal(str_order(string, decreasing=True), np.array([0, 1, 2]))

    def test_strOrder_series(self):
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(str_order(string), pd.Series([0, 1, 3, 2]))

    def test_strOrder_natural(self):
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(str_order(string, numeric=True), pd.Series([3, 2, 1, 0]))

    def test_strOrder_none(self):
        string = ['a', 'b', 'c', None]
        assert str_order(string, na_last=False) == [0, 3, 1, 2]

    # String Detect
    def test_strDetect_string(self):
        string = 'video'
        assert str_detect(string, '[aeiou]') is True

    def test_strDetect_list(self):
        x = ["why", "video", "cross", "extra", "deal", "authority"]
        assert str_detect(x, '[aeiou]') == [False, True, True, True, True, True]

    def test_strDetect_array(self):
        x = np.array(["why", "video", "cross", "extra", "deal", "authority"])
        np.testing.assert_array_equal(str_detect(x, '[aeiou]', negate=True), np.array([True, False, False, False, False, False]))

    def test_strDetect_series(self):
        x = pd.Series(["why", "video", "cross", "extra", "deal", "authority"])
        pd.testing.assert_series_equal(str_detect(x, '[aeiou]'), pd.Series([False, True, True, True, True, True]))

    # String Count
    def test_strCount_string(self):
        string = 'video'
        assert str_count(string, '[aeiou]') == 3

    def test_strCount_list(self):
        fruit = ["apple", "banana", "pear", "pineapple"]
        assert str_count(fruit, 'a') == [1, 3, 1, 1]

    def test_strCount_array(self):
        fruit = np.array(["apple", "banana", "pear", "pineapple"])
        np.testing.assert_array_equal(str_count(fruit, 'e'), np.array([1, 0, 1, 2]))

    def test_strCount_series(self):
        fruit = pd.Series(["apple", "banana", "pear", "pineapple"])
        pd.testing.assert_series_equal(str_count(fruit, ['a', 'b', 'p', 'p']), pd.Series([1, 1, 1, 3]))

    # String Subset
    def test_strSubset_string(self):
        string = 'video'
        assert str_subset(string, '[aeiou]') == 'video'

    def test_strSubset_list(self):
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        assert str_subset(string, '[aeiou]') == ['video', 'cross', 'extra', 'deal', 'authority']

    def test_strSubset_array(self):
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        np.testing.assert_array_equal(str_subset(string, '[aeiou]', negate=True), np.array(['why']))

    def test_strSubset_series(self):
        string = pd.Series(["why", "video", "cross", "extra", "deal", "authority"])
        expected = pd.Series(['video', 'cross', 'authority'])
        expected.index = [1, 2, 5]
        pd.testing.assert_series_equal(str_subset(string, 'o'), expected)


if __name__ == '__main__':
    unittest.main()