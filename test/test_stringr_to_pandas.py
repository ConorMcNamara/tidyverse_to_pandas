import unittest
import pytest
import pandas as pd
import numpy as np
from src.stringr_to_pandas import str_length, str_sub, str_detect, str_count, str_dup, str_subset, str_to_upper, \
    str_to_lower, str_to_sentence, str_to_title, str_replace_all, str_order, str_sort, str_flatten, str_trunc, \
    str_remove_all, str_replace_na, str_replace, str_remove, str_split, str_split_fixed, str_split_n, str_pad, \
    str_squish, str_trim, str_which, str_starts, str_ends, str_extract, str_extract_all, str_match, str_match_all, \
    str_equal, str_unique


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

    # String Replace NA
    def test_strReplaceNA_string(self):
        string = np.nan
        assert str_replace_na(string) == 'NA'

    def test_strReplaceNA_list(self):
        string = [np.nan, None, 'poop']
        assert str_replace_na(string) == ['NA', 'NA', 'poop']

    def test_strReplaceNA_array(self):
        string = np.array(['foo', 'bar', None, np.nan])
        np.testing.assert_array_equal(str_replace_na(string), ['foo', 'bar', 'NA', 'NA'])

    def test_strReplaceNA_series(self):
        string = pd.Series(['Mass', None, 'Effect', np.nan])
        pd.testing.assert_series_equal(str_replace_na(string), pd.Series(['Mass', 'NA', 'Effect', 'NA']))

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
        assert str_order(string, na_last=False) == [3, 0, 1, 2]

    # String Sort
    def test_strSort_string(self):
        string = 'string'
        assert str_sort(string) == string

    def test_strSort_list(self):
        string = ['i', 'am', 'a', 'string']
        assert str_sort(string) == ['a', 'am', 'i', 'string']

    def test_strSort_array(self):
        string = np.array(['this', 'is', 'happening'])
        np.testing.assert_array_equal(str_sort(string, decreasing=True), np.array(["this", 'is', 'happening']))

    def test_strSort_series(self):
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(str_sort(string), pd.Series(["100a10", "100a5", "2a", "2b"], index=[0, 1, 3, 2]))

    def test_strSort_natural(self):
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(str_sort(string, numeric=True), pd.Series(["2a", '2b', '100a5', '100a10'], index=[3, 2, 1, 0]))

    def test_strSort_none(self):
        string = ['a', 'b', 'c', None]
        assert str_sort(string, na_last=False) == [None, 'a', 'b', 'c']

    # String Pad
    def test_strPad_string(self):
        assert str_pad("halsey", 30, "left") == "                        halsey"

    def test_strPad_list(self):
        string = ["a", "abc", "abcdef"]
        assert str_pad(string, 10) == ["a         ", "abc       ", "abcdef    "]

    def test_strPad_array(self):
        string = np.array(["a", "a", "a"])
        np.testing.assert_array_equal(str_pad(string, [5, 10, 20], 'left'), np.array(["    a", "         a", "                   a"]))

    def test_strPad_series(self):
        string = pd.Series(["a", "a", "a"])
        pd.testing.assert_series_equal(str_pad(string, 10, 'left', pad=["-", "_", " "]), pd.Series(["---------a", "_________a", "         a"]))

    # String Trim
    def test_strTrim_string(self):
        string = "  String with trailing and leading white space\t"
        assert str_trim(string, "both") == "String with trailing and leading white space"

    def test_strTrim_list(self):
        string = ["\n\nString with trailing and leading white space\n\n", "  String with trailing and leading white space\t"]
        assert str_trim(string, "both") == ["String with trailing and leading white space", "String with trailing and leading white space"]

    def test_strTrim_array(self):
        string = np.array(['\n\nString with trailing and leading white space\n\n', "  String with trailing and leading white space\t"])
        np.testing.assert_array_equal(str_trim(string, "right"), np.array(["String with trailing and leading white space\n\n", "String with trailing and leading white space\t"]))

    def test_strTrime_series(self):
        string = pd.Series(['\n\nString with trailing and leading white space\n\n', "  String with trailing and leading white space\t"])
        pd.testing.assert_series_equal(str_trim(string, "left"), pd.Series(['\n\nString with trailing and leading white space', '  String with trailing and leading white space']))

    # String Squish
    def test_strSquish_string(self):
        string = "  String with trailing,  middle, and leading white space\t"
        assert str_squish(string) == "String with trailing, middle, and leading white space"

    def test_strSquish_list(self):
        string = ["\n\nString with excess,  trailing and leading white   space\n\n", "  String with trailing,  middle, and leading white space\t"]
        assert str_squish(string) == ["String with excess, trailing and leading white space", "String with trailing, middle, and leading white space"]

    def test_strSquish_array(self):
        string = np.array(["  String with trailing   and   leading white space\t", "  String with trailing,  middle, and leading white space\t"])
        np.testing.assert_array_equal(str_squish(string), np.array(["String with trailing and leading white space", "String with trailing, middle, and leading white space"]))

    def test_strSquish_series(self):
        string = pd.Series(["\n\nString with excess,  trailing and leading white   space\n\n", "  String with trailing   and   leading white space\t"])
        pd.testing.assert_series_equal(str_squish(string), pd.Series(["String with excess, trailing and leading white space", "String with trailing and leading white space"]))

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

    # String Which
    def test_strWhich_string(self):
        string = 'video'
        assert str_which(string, '[aeiou]') == 0

    def test_strWhich_list(self):
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        assert str_which(string, '[aeiou]') == [1, 2, 3, 4, 5]

    def test_strWhich_array(self):
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        np.testing.assert_array_equal(str_which(string, '[aeiou]', negate=True), np.array([0]))

    def test_strWhich_series(self):
        string = pd.Series(["why", "video", "cross", "extra", "deal", "authority"])
        expected = pd.Series([1, 2, 5])
        pd.testing.assert_series_equal(str_which(string, 'o'), expected)

    # String Replace
    def test_strReplace_string(self):
        fruits = "one apple"
        assert str_replace(fruits, "[aeiou]", "-") == '-ne apple'

    def test_strReplace_list(self):
        fruits = ["one apple", "two pears", "three bananas"]
        assert str_replace(fruits, "[aeiou]", "-") == ["-ne apple", "tw- pears", "thr-e bananas"]

    def test_strReplace_array(self):
        fruits = np.array(["one apple", "two pears", "three bananas"])
        np.testing.assert_array_equal(str_replace(fruits, "b", np.nan), np.array(["one apple", "two pears", np.nan]))

    def test_strReplace_series(self):
        fruits = pd.Series(["one apple", "two pears", "three bananas"])
        expected = pd.Series(['1 apple', '2 pears', '3 bananas'])
        pd.testing.assert_series_equal(str_replace(fruits, ['one', 'two', 'three'], ['1', '2', '3']), expected)

    # String Replace All
    def test_strReplaceAll_string(self):
        fruits ="one apple"
        assert str_replace_all(fruits, "[aeiou]", "-") == '-n- -ppl-'

    def test_strReplaceAll_list(self):
        fruits = ["one apple", "two pears", "three bananas"]
        assert str_replace_all(fruits, "[aeiou]", "-") == ["-n- -ppl-", "tw- p--rs", "thr-- b-n-n-s"]

    def test_strReplaceAll_array(self):
        fruits = np.array(["one apple", "two pears", "three bananas"])
        np.testing.assert_array_equal(str_replace_all(fruits, "b", np.nan), np.array(["one apple", "two pears", np.nan]))

    def test_strReplaceAll_series(self):
        fruits = pd.Series(["one apple", "two pears", "three bananas"])
        expected = pd.Series(['1 apple', '2 pears', '3 bananas'])
        pd.testing.assert_series_equal(str_replace_all(fruits, ['one', 'two', 'three'], ['1', '2', '3']), expected)

    # String Remove
    def test_strRemove_string(self):
        fruits = "one apple"
        assert str_remove(fruits, '[aeiou]') == "ne apple"

    def test_strRemove_list(self):
        fruits = ["one apple", "two pears", "three bananas"]
        assert str_remove(fruits, "[aeiou]") == ["ne apple", "tw pears", "thre bananas"]

    def test_strRemove_array(self):
        relient_k = np.array(['Devastation and Reform', 'I Need You', 'The Best Thing'])
        expected = np.array(['Devastation Reform', 'I Need You', 'The Best Thing'])
        np.testing.assert_array_equal(str_remove(relient_k, "and "), expected)

    def test_strRemove_series(self):
        lyrics = pd.Series(['Come', 'Right', 'Out', 'and', 'Say', 'It'])
        expected = pd.Series(['Come', 'Rght', 'Out', 'and', 'Say', 't'])
        pd.testing.assert_series_equal(str_remove(lyrics, '[iI]'), expected)

    # String Remove All
    def test_strRemoveAll_string(self):
        fruits = "one apple"
        assert str_remove_all(fruits, '[aeiou]') == "n ppl"

    def test_strRemoveAll_list(self):
        fruits = ["one apple", "two pears", "three bananas"]
        assert str_remove_all(fruits, "[aeiou]") == ["n ppl", "tw prs", "thr bnns"]

    def test_strRemoveAll_array(self):
        relient_k = np.array(['Devastation and Reform', 'I Need You', 'The Best Thing'])
        expected = np.array(['Devastation Reform', 'I Need You', 'The Best Thing'])
        np.testing.assert_array_equal(str_remove_all(relient_k, "and "), expected)

    def test_strRemoveAll_series(self):
        lyrics = pd.Series(['Come', 'Right', 'Out', 'and', 'Say', 'It'])
        expected = pd.Series(['Come', 'Rght', 'Out', 'and', 'Say', 't'])
        pd.testing.assert_series_equal(str_remove_all(lyrics, '[iI]'), expected)

    # String Split
    def test_strSplit_string(self):
        string = "guinea pig and farts"
        assert str_split(string, " and ") == ['guinea pig', 'farts']

    def test_strSplit_list(self):
        fruits = ["apples and oranges and pears and bananas", "pineapples and mangos and guavas"]
        expected = [['apples', 'oranges', 'pears', 'bananas'], ['pineapples', 'mangos', 'guavas', '']]
        assert str_split(fruits, " and ", simplify=True) == expected

    def test_strSplit_array(self):
        fruits = np.array(["apples and oranges and pears and bananas", "pineapples and mangos and guavas"])
        expected = np.array([["apples", "oranges and pears and bananas"], ["pineapples", "mangos and guavas"]])
        assert all(str_split(fruits, " and ", n=1)[0] == expected[0])
        assert all(str_split(fruits, " and ", n=1)[1] == expected[1])

    def test_strSplit_series(self):
        fruits = pd.Series(["apples and oranges and pears and bananas", "pineapples and mangos and guavas"])
        actual = str_split(fruits, " and ", n=5)
        expected = pd.Series([['apples', 'oranges', 'pears', 'bananas'], ['pineapples', 'mangos', 'guavas']])
        pd.testing.assert_series_equal(actual, expected)

    # String Split Fixed
    def test_strSplitFixed_string(self):
        string = "guinea pig and farts"
        assert str_split_fixed(string, " and ") == ['guinea pig', 'farts']

    def test_strSplitFixed_list(self):
        fruits = ["apples and oranges and pears and bananas", "pineapples and mangos and guavas"]
        expected = [["apples", "oranges", "pears and bananas"], ["pineapples", "mangos", "guavas"]]
        assert str_split_fixed(fruits, " and ", 2) == expected

    def test_strSplitFixed_array(self):
        fruits = np.array(["apples and oranges and pears and bananas", "pineapples and mangos and guavas"])
        expected = np.array([['apples', 'oranges and pears and bananas'], ["pineapples", "mangos and guavas"]])
        np.testing.assert_array_equal(str_split_fixed(fruits, " and ", n=1), expected)

    def test_strSplitFixed_series(self):
        fruits = pd.Series(["apples and oranges and pears and bananas", "pineapples and mangos and guavas"])
        expected = pd.DataFrame([['apples', 'oranges', 'pears', 'bananas'], ['pineapples', 'mangos', 'guavas', ""]])
        pd.testing.assert_frame_equal(str_split_fixed(fruits, " and ", 3), expected)

    # String Split N
    def test_strSplitN_string(self):
        string = 'guinea pigs and farts'
        assert str_split_n(string, " ", 1) == 'pigs'

    def test_strSplitN_list(self):
        fruits = ["apples and oranges and pears and bananas", "pineapples and mangos and guavas"]
        assert str_split_n(fruits, " and ", 0) == ['apples', 'pineapples']

    def test_strSplitN_array(self):
        fruits = np.array(["apples and oranges and pears and bananas", "pineapples and mangos and guavas"])
        assert str_split_n(fruits, " and ", 3)[0] == 'bananas'
        assert str_split_n(fruits, " and ", 3)[1] is np.nan

    def test_strSplitN_series(self):
        fruits = pd.Series(["apples and oranges and pears and bananas", "pineapples and mangos and guavas"])
        pd.testing.assert_series_equal(str_split_n(fruits, " and ", 3), pd.Series(['bananas', np.nan], name=3))

    # String Starts
    def test_strStarts_string(self):
        string = 'guinea pigs and farts'
        assert str_starts(string, 'guinea') == True

    def test_strStarts_list(self):
        fruit = ["apple", "banana", "pear", "pineapple"]
        assert str_starts(fruit, "p") == [False, False, True, True]

    def test_strStarts_array(self):
        fruit = np.array(["apple", "banana", "pear", "pineapple"])
        np.testing.assert_array_equal(str_starts(fruit, "p", negate=True), np.array([True, True, False, False]))

    def test_strStarts_series(self):
        fruit = pd.Series(["apple", "banana", "pear", "pineapple"])
        pd.testing.assert_series_equal(str_starts(fruit, '[aeiou]'), pd.Series([True, False, False, False]))

    # String Ends
    def test_strEnds_string(self):
        string = 'guinea pigs and farts'
        assert str_ends(string, 'farts') == True

    def test_strEnds_list(self):
        fruit = ["apple", "banana", "pear", "pineapple"]
        assert str_ends(fruit, "e") == [True, False, False, True]

    def test_strEnds_array(self):
        fruit = np.array(["apple", "banana", "pear", "pineapple"])
        np.testing.assert_array_equal(str_ends(fruit, 'e', True), np.array([False, True, True, False]))

    def test_strEnds_series(self):
        fruit = pd.Series(["apple", "banana", "pear", "pineapple"])
        pd.testing.assert_series_equal(str_ends(fruit, '[aeiou]'), pd.Series([True, True, False, True]))

    # String Extract
    def test_strExtract_string(self):
        number = "219 733 8965"
        assert str_extract(number, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == number

    def test_strExtract_list(self):
        shopping_list = ["apples x4", "bag of flour", "bag of sugar", "milk x2"]
        assert str_extract(shopping_list, "\\d") == ['4', None, None, '2']

    def test_strExtract_array(self):
        shopping_list = np.array(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        np.testing.assert_array_equal(str_extract(shopping_list, "[a-z]{1,4}"), np.array(["appl", "bag", "bag", "milk"]))

    def test_strExtract_series(self):
        shopping_list = pd.Series(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        pd.testing.assert_series_equal(str_extract(shopping_list, "[a-z]+"), pd.Series(['apples', 'bag', 'bag', 'milk']))

    # String Extract All
    def test_strExtractAll_string(self):
        number = "219 733 8965"
        assert str_extract_all(number, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == [('219', '733', '8965')]

    def test_strExtractAll_list(self):
        shopping_list = ["apples x4", "bag of flour", "bag of sugar", "milk x2"]
        assert str_extract_all(shopping_list, "\\b[a-z]+\\b") == [['apples'], ['bag', 'of', 'flour'], ['bag', 'of', 'sugar'], ['milk']]

    def test_strExtractAll_array(self):
        shopping_list = np.array(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        np.testing.assert_array_equal(str_extract_all(shopping_list, "\\b[a-z]+\\b", simplify=True),
                                      np.array([['apples', '', ''], ['bag', 'of', 'flour'], ['bag', 'of', 'sugar'], ['milk', '', '']]))

    def test_strExtractAll_series(self):
        shopping_list = pd.Series(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        pd.testing.assert_series_equal(str_extract_all(shopping_list, "\\d", simplify=True), pd.Series(["4", "", "", "2"], name='match'))

    # String Match
    def test_strMatch_string(self):
        string = "<a> <b>"
        assert str_match(string, pattern="<(.*?)> <(.*?)>") == ["<a> <b>", "a", "b"]

    def test_strMatch_list(self):
        strings = [" 219 733 8965", "329-293-8753 ", "banana", "595 794 7569", "387 287 6718", "apple", "233.398.9187  ",
                   "482 952 3315", "239 923 8115 and 842 566 4692", "Work: 579-499-7527", "$1000", "Home: 543.355.3679"]
        expected = [['219 733 8965', '219', '733', '8965'], ['329-293-8753', '329', '293', '8753'], [None, None, None, None],
                    ['595 794 7569', '595', '794', '7569'], ['387 287 6718', '387', '287', '6718'], [None, None, None, None],
                    ['233.398.9187', '233', '398', '9187'], ['482 952 3315', '482', '952', '3315'],['239 923 8115', '239', '923', '8115'],
                    ['579-499-7527', '579', '499', '7527'], [None, None, None, None], ['543.355.3679', '543', '355', '3679']]
        assert str_match(strings, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == expected

    def test_strMatch_array(self):
        string = np.array(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = np.array([['<a> <b>', 'a', 'b'], ['<a> <>', 'a', ''], [None, None, None], [None, None, None], [None, None, None]])
        np.testing.assert_array_equal(str_match(string, "<(.*?)> <(.*?)>"), expected)

    def test_strMatch_series(self):
        string = pd.Series(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = pd.DataFrame({'whole_match': ['<a> <b>', '<a> <>', None, None, None],
                              'match': ['a', 'a', np.nan, np.nan, np.nan]})
        pd.testing.assert_frame_equal(str_match(string, "<(.*?)> <(.*?)>"), expected)

    # String Match All
    def test_strMatchAll_string(self):
        string = "239 923 8115 and 842 566 4692"
        expected = [["239 923 8115", "239", "923", "8115"], ["842 566 4692", "842", "566", "4692"]]
        assert str_match_all(string, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == expected

    def test_strMatchAll_list(self):
        strings = [" 219 733 8965", "329-293-8753 ", "banana", "595 794 7569", "387 287 6718", "apple", "233.398.9187  ",
                   "482 952 3315", "239 923 8115 and 842 566 4692", "Work: 579-499-7527", "$1000", "Home: 543.355.3679"]
        expected = [['219 733 8965', '219', '733', '8965'], ['329-293-8753', '329', '293', '8753'], ['', '', '', ''],
                    ['595 794 7569', '595', '794', '7569'], ['387 287 6718', '387', '287', '6718'], ['', '', '', ''],
                    ['233.398.9187', '233', '398', '9187'], ['482 952 3315', '482', '952', '3315'], ['239 923 8115', '239', '923', '8115'],
                    ['842 566 4692', '842', '566', '4692'], ['579-499-7527', '579', '499', '7527'], ['', '', '', ''],
                    ['543.355.3679', '543', '355', '3679']]
        assert str_match_all(strings, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == expected

    def test_strMatchAll_array(self):
        x = np.array(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = np.array([['<a>', 'a'], ['<b>', 'b'], ['<a>', 'a'], ['', ''], ['<a>', 'a'], ['', ''], ['', '']])
        np.testing.assert_array_equal(str_match_all(x, "<(.*?)>"), expected)

    def test_strMatchAll_series(self):
        x = pd.Series(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = pd.DataFrame([['<a>', 'a'], ['<b>', 'b'], ['<a>', 'a'], ['<>', ''], ['<a>', 'a'], ['', ''], ['', '']])
        expected.columns = ['whole_match', 0]
        pd.testing.assert_frame_equal(str_match_all(x, "<(.*?)>"), expected)
        
# String Equal
class TestStrEqual:

    @staticmethod
    def test_strEqual_str():
        string1 = '\u03A9'
        string2 = '\u2126'
        assert stp.str_equal(string1, string2) is True
        assert stp.str_equal(string1, string2, True) is True

    @staticmethod
    def test_strEqual_list():
        string1 = ['\u00e1']
        string2 = ["a\u0301"]
        np.testing.assert_array_equal(stp.str_equal(string1, string2, True), [True])
        np.testing.assert_array_equal(stp.str_equal(string1, string2), [True])

    @staticmethod
    def test_strEqual_array():
        string1 = np.array(['guinea', 'pigs', 'and', 'farts'])
        string2 = np.array(['Guinea', 'pigs', '&', 'farts'])
        np.testing.assert_array_equal(stp.str_equal(string1, string2), np.array([False, True, False, True]))
        np.testing.assert_array_equal(stp.str_equal(string1, string2, True), np.array([True, True, False, True]))

    @staticmethod
    def test_strEqual_series():
        string1 = pd.Series(["Farts", "and", "guinea", "pigs"])
        string2 = pd.Series(["farts", "&", "guinea", "pigs"])
        pd.testing.assert_series_equal(pd.Series([True, False, True, True]), stp.str_equal(string1, string2, True))
        pd.testing.assert_series_equal(pd.Series([False, False, True, True]), stp.str_equal(string1, string2))

        
class TestStrUnique:

    @staticmethod
    def test_strUnique_string():
        string = "guinea pig"
        assert stp.str_unique(string) == "guinea p"
        assert stp.str_unique(string, True) == "guinea p"

    @staticmethod
    def test_strUnique_list():
        string = ["a", "b", "c", "b", "a"]
        assert stp.str_unique(string, 1) == ['a', 'b', 'c']
        assert stp.str_unique(string, 4) == ['a', 'b', 'c']

    @staticmethod
    def test_strUnique_array():
        string = np.array(["motley", "mötley", "pinguino", "pingüino"])
        np.testing.assert_array_equal(stp.str_unique(string, 1), np.array(['motley', 'pinguino']))
        np.testing.assert_array_equal(stp.str_unique(string, 4), string)

    @staticmethod
    def test_strUnique_series():
        string = pd.Series(["What", "what", "is", "love"])
        pd.testing.assert_series_equal(stp.str_unique(string, 1), pd.Series(["What", "is", "love"], index=[0, 1, 2]))
        pd.testing.assert_series_equal(stp.str_unique(string, 4), string)
 

if __name__ == '__main__':
    unittest.main()
