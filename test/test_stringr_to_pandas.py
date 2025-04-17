import pytest

import numpy as np
import pandas as pd

import tidyverse.stringr_to_pandas as stp


class TestStrLength:
    # String Length
    @staticmethod
    def test_strLength_string() -> None:
        string = "abcd"
        assert stp.str_length(string) == 4

    @staticmethod
    def test_strLength_list() -> None:
        string = ["ab", "cdef"]
        assert stp.str_length(string) == [2, 4]

    @staticmethod
    def test_strLength_array() -> None:
        string = np.array(["abc", "defg"])
        np.testing.assert_array_equal(stp.str_length(string), np.array([3, 4]))

    @staticmethod
    def test_strLength_series() -> None:
        string = pd.Series(["abcd", "efg"])
        pd.testing.assert_series_equal(stp.str_length(string), pd.Series([4, 3]))


class TestStrSub:
    # String Sub
    @staticmethod
    def test_strSub_string() -> None:
        string = "abcd"
        assert stp.str_sub(string, 0, 3) == "abc"

    @staticmethod
    def test_strSub_list() -> None:
        string = ["abcdef", "ghifjk"]
        assert stp.str_sub(string, 2, 3) == ["c", "i"]

    @staticmethod
    def test_strSub_array() -> None:
        string = np.array(["abcdef", "ghifjk"])
        np.testing.assert_array_equal(stp.str_sub(string, 1, -1), np.array(["bcde", "hifj"]))

    @staticmethod
    def test_strSub_series() -> None:
        string = pd.Series(["abcdef", "ghifjk"])
        pd.testing.assert_series_equal(stp.str_sub(string, 2, -1), pd.Series(["cde", "ifj"]))


class TestStrDup:
    # String Duplicate
    @staticmethod
    def test_strDup_string() -> None:
        string = "abcd"
        assert stp.str_dup(string, 2) == "abcdabcd"

    @staticmethod
    def test_strDup_list() -> None:
        string = ["aa", "bb"]
        assert stp.str_dup(string, 2) == ["aaaa", "bbbb"]

    @staticmethod
    def test_strDup_array() -> None:
        string = np.array(["zz", "yy"])
        np.testing.assert_array_equal(stp.str_dup(string, [2, 3]), np.array(["zzzz", "yyyyyy"]))

    @staticmethod
    def test_strDup_series() -> None:
        string = pd.Series(["abc", "xyz"])
        pd.testing.assert_series_equal(stp.str_dup(string, 3), pd.Series(["abcabcabc", "xyzxyzxyz"]))


class TestStrFlatten:
    # String Flatten
    @staticmethod
    def test_strFlatten_string() -> None:
        string = "Mass Effect: Andromeda"
        assert stp.str_flatten(string, ",") == string

    @staticmethod
    def test_strFlatten_list() -> None:
        string = ["This", "is", "a", "test"]
        assert stp.str_flatten(string, " ") == "This is a test"

    @staticmethod
    def test_strFlatten_array() -> None:
        string = np.array(["The", "only", "thing", "they", "fear", "is", "you"])
        assert stp.str_flatten(string, ".") == "The.only.thing.they.fear.is.you"

    @staticmethod
    def test_strFlatten_series() -> None:
        string = pd.Series(["A", "B", "B", "A"])
        assert stp.str_flatten(string) == "ABBA"


class TestStrTrunc:
    # String Truncate
    @staticmethod
    def test_strTrunc_string() -> None:
        string = "Mass Effect: Andromeda"
        assert stp.str_trunc(string, 11, "right") == "Mass Effect..."

    @staticmethod
    def test_strTrunc_list() -> None:
        string = ["This  string  is  moderately  long", "Guinea  pigs  and  farts"]
        assert stp.str_trunc(string, 20, "center") == [
            "This  stri...tely  long",
            "Guinea  pi...and  farts",
        ]

    @staticmethod
    def test_strTrunc_array() -> None:
        string = np.array(["This  string  is  moderately  long", "Guinea  pigs  and  farts"])
        expected = np.array(["...is  moderately  long", "...ea  pigs  and  farts"])
        np.testing.assert_array_equal(stp.str_trunc(string, 20, "left"), expected)

    @staticmethod
    def test_strTrunc_series() -> None:
        string = pd.Series(["This  string  is  moderately  long", "Guinea  pigs  and  farts"])
        expected = pd.Series(["This  string  is  mo...", "Guinea  pigs  and  f..."])
        pd.testing.assert_series_equal(stp.str_trunc(string, 20), expected)


class TestStrReplaceNA:
    # String Replace NA
    @staticmethod
    def test_strReplaceNA_string() -> None:
        string = np.nan
        assert stp.str_replace_na(string) == "NA"

    @staticmethod
    def test_strReplaceNA_list() -> None:
        string = [np.nan, None, "poop"]
        assert stp.str_replace_na(string) == ["NA", "NA", "poop"]

    @staticmethod
    def test_strReplaceNA_array() -> None:
        string = np.array(["foo", "bar", None, np.nan])
        np.testing.assert_array_equal(stp.str_replace_na(string), ["foo", "bar", "NA", "NA"])

    @staticmethod
    def test_strReplaceNA_series() -> None:
        string = pd.Series(["Mass", None, "Effect", np.nan])
        pd.testing.assert_series_equal(stp.str_replace_na(string), pd.Series(["Mass", "NA", "Effect", "NA"]))


class TestStrToUpper:
    # String Uppercase
    @staticmethod
    def test_strToUpper_string() -> None:
        string = "aabb"
        assert stp.str_to_upper(string) == "AABB"

    @staticmethod
    def test_strToUpper_list() -> None:
        string = ["a", "ab", "abc"]
        assert stp.str_to_upper(string) == ["A", "AB", "ABC"]

    @staticmethod
    def test_strToUpper_array() -> None:
        string = np.array(["video", "killed", "radio", "star"])
        np.testing.assert_array_equal(stp.str_to_upper(string), np.array(["VIDEO", "KILLED", "RADIO", "STAR"]))

    @staticmethod
    def test_strToUpper_series() -> None:
        string = pd.Series(["green", "lantern", "first", "flight"])
        pd.testing.assert_series_equal(stp.str_to_upper(string), pd.Series(["GREEN", "LANTERN", "FIRST", "FLIGHT"]))


class TestStrToLower:
    # String Lowercase
    @staticmethod
    def test_strToLower_string() -> None:
        string = "ABBA"
        assert stp.str_to_lower(string) == "abba"

    @staticmethod
    def test_strToLower_list() -> None:
        string = ["DANCING", "QUEEN"]
        assert stp.str_to_lower(string) == ["dancing", "queen"]

    @staticmethod
    def test_strToLower_array() -> None:
        string = np.array(["I", "AM", "THE", "LAW"])
        np.testing.assert_array_equal(stp.str_to_lower(string), np.array(["i", "am", "the", "law"]))

    @staticmethod
    def test_strToLower_series() -> None:
        string = pd.Series(["WHY", "IS", "THIS", "HAPPENING"])
        pd.testing.assert_series_equal(stp.str_to_lower(string), pd.Series(["why", "is", "this", "happening"]))


class StrToTitle:
    # String To Title
    @staticmethod
    def test_strToTitle_string() -> None:
        string = "what is going on here?"
        assert stp.str_to_title(string) == "What Is Going On Here?"

    @staticmethod
    def test_strToTitle_list() -> None:
        string = ["i hate this", "so much"]
        assert stp.str_to_title(string) == ["I Hate This", "So Much"]

    @staticmethod
    def test_strToTitle_array() -> None:
        string = np.array(["the", "flight", "of", "fear"])
        np.testing.assert_array_equal(stp.str_to_title(string), np.array(["The", "Flight", "Of", "Fear"]))

    @staticmethod
    def test_strToTitle_series() -> None:
        string = pd.Series(["guinea pigs", "and farts"])
        pd.testing.assert_series_equal(stp.str_to_title(string), pd.Series(["Guinea Pigs", "And Farts"]))


class StrToSentence:
    # String To Sentence
    @staticmethod
    def test_strToSentence_string() -> None:
        string = "this is a sentence"
        assert stp.str_to_sentence(string) == "This is a sentence"

    @staticmethod
    def test_strToSentence_list() -> None:
        string = ["michael eisner", "roy disney"]
        assert stp.str_to_sentence(string) == ["Michael eisner", "Roy disney"]

    @staticmethod
    def test_strToSentence_array() -> None:
        string = np.array(["what is love", "baby don't hurt me"])
        np.testing.assert_array_equal(
            stp.str_to_sentence(string),
            np.array(["What is love", "Baby don't hurt me"]),
        )

    @staticmethod
    def test_strToSentence_series() -> None:
        string = pd.Series(["that's the sound your mother made", "lasht night"])
        pd.testing.assert_series_equal(
            stp.str_to_sentence(string),
            pd.Series(["That's the sound your mother made", "Lasht night"]),
        )


class TestStrOrder:
    # String Order
    @staticmethod
    def test_strOrder_string() -> None:
        string = "string"
        assert stp.str_order(string) == 0

    @staticmethod
    def test_strOrder_list() -> None:
        string = ["i", "am", "a", "string"]
        assert stp.str_order(string) == [2, 1, 0, 3]

    @staticmethod
    def test_strOrder_array() -> None:
        string = np.array(["this", "is", "happening"])
        np.testing.assert_array_equal(stp.str_order(string, decreasing=True), np.array([0, 1, 2]))

    @staticmethod
    def test_strOrder_series() -> None:
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(stp.str_order(string), pd.Series([0, 1, 3, 2]))

    @staticmethod
    def test_strOrder_natural() -> None:
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(stp.str_order(string, numeric=True), pd.Series([3, 2, 1, 0]))

    @staticmethod
    def test_strOrder_none() -> None:
        string = ["a", "b", "c", None]
        assert stp.str_order(string, na_last=False) == [3, 0, 1, 2]


class TestStrSort:
    # String Sort
    @staticmethod
    def test_strSort_string() -> None:
        string = "string"
        assert stp.str_sort(string) == string

    @staticmethod
    def test_strSort_list() -> None:
        string = ["i", "am", "a", "string"]
        assert stp.str_sort(string) == ["a", "am", "i", "string"]

    @staticmethod
    def test_strSort_array() -> None:
        string = np.array(["this", "is", "happening"])
        np.testing.assert_array_equal(stp.str_sort(string, decreasing=True), np.array(["this", "is", "happening"]))

    @staticmethod
    def test_strSort_series() -> None:
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(
            stp.str_sort(string),
            pd.Series(["100a10", "100a5", "2a", "2b"], index=[0, 1, 3, 2]),
        )

    @staticmethod
    def test_strSort_natural() -> None:
        string = pd.Series(["100a10", "100a5", "2b", "2a"])
        pd.testing.assert_series_equal(
            stp.str_sort(string, numeric=True),
            pd.Series(["2a", "2b", "100a5", "100a10"], index=[3, 2, 1, 0]),
        )

    @staticmethod
    def test_strSort_none() -> None:
        string = ["a", "b", "c", None]
        assert stp.str_sort(string, na_last=False) == [None, "a", "b", "c"]


class StrPad:
    # String Pad
    @staticmethod
    def test_strPad_string() -> None:
        assert stp.str_pad("halsey", 30, "left") == "                        halsey"

    @staticmethod
    def test_strPad_list() -> None:
        string = ["a", "abc", "abcdef"]
        assert stp.str_pad(string, 10) == ["a         ", "abc       ", "abcdef    "]

    @staticmethod
    def test_strPad_array() -> None:
        string = np.array(["a", "a", "a"])
        np.testing.assert_array_equal(
            stp.str_pad(string, [5, 10, 20], "left"),
            np.array(["    a", "         a", "                   a"]),
        )

    @staticmethod
    def test_strPad_series() -> None:
        string = pd.Series(["a", "a", "a"])
        pd.testing.assert_series_equal(
            stp.str_pad(string, 10, "left", pad=["-", "_", " "]),
            pd.Series(["---------a", "_________a", "         a"]),
        )


class TestStrTrim:
    # String Trim
    @staticmethod
    def test_strTrim_string() -> None:
        string = "  String with trailing and leading white space\t"
        assert stp.str_trim(string, "both") == "String with trailing and leading white space"

    @staticmethod
    def test_strTrim_list() -> None:
        string = [
            "\n\nString with trailing and leading white space\n\n",
            "  String with trailing and leading white space\t",
        ]
        assert stp.str_trim(string, "both") == [
            "String with trailing and leading white space",
            "String with trailing and leading white space",
        ]

    @staticmethod
    def test_strTrim_array() -> None:
        string = np.array(
            [
                "\n\nString with trailing and leading white space\n\n",
                "  String with trailing and leading white space\t",
            ]
        )
        np.testing.assert_array_equal(
            stp.str_trim(string, "right"),
            np.array(
                [
                    "String with trailing and leading white space\n\n",
                    "String with trailing and leading white space\t",
                ]
            ),
        )

    @staticmethod
    def test_strTrime_series() -> None:
        string = pd.Series(
            [
                "\n\nString with trailing and leading white space\n\n",
                "  String with trailing and leading white space\t",
            ]
        )
        pd.testing.assert_series_equal(
            stp.str_trim(string, "left"),
            pd.Series(
                [
                    "\n\nString with trailing and leading white space",
                    "  String with trailing and leading white space",
                ]
            ),
        )


class StrSquish:
    # String Squish
    @staticmethod
    def test_strSquish_string() -> None:
        string = "  String with trailing,  middle, and leading white space\t"
        assert stp.str_squish(string) == "String with trailing, middle, and leading white space"

    @staticmethod
    def test_strSquish_list() -> None:
        string = [
            "\n\nString with excess,  trailing and leading white   space\n\n",
            "  String with trailing,  middle, and leading white space\t",
        ]
        assert stp.str_squish(string) == [
            "String with excess, trailing and leading white space",
            "String with trailing, middle, and leading white space",
        ]

    @staticmethod
    def test_strSquish_array() -> None:
        string = np.array(
            [
                "  String with trailing   and   leading white space\t",
                "  String with trailing,  middle, and leading white space\t",
            ]
        )
        np.testing.assert_array_equal(
            stp.str_squish(string),
            np.array(
                [
                    "String with trailing and leading white space",
                    "String with trailing, middle, and leading white space",
                ]
            ),
        )

    @staticmethod
    def test_strSquish_series() -> None:
        string = pd.Series(
            [
                "\n\nString with excess,  trailing and leading white   space\n\n",
                "  String with trailing   and   leading white space\t",
            ]
        )
        pd.testing.assert_series_equal(
            stp.str_squish(string),
            pd.Series(
                [
                    "String with excess, trailing and leading white space",
                    "String with trailing and leading white space",
                ]
            ),
        )


class StrDefect:
    # String Detect
    @staticmethod
    def test_strDetect_string() -> None:
        string = "video"
        assert stp.str_detect(string, "[aeiou]") is True

    @staticmethod
    def test_strDetect_list() -> None:
        x = ["why", "video", "cross", "extra", "deal", "authority"]
        assert stp.str_detect(x, "[aeiou]") == [False, True, True, True, True, True]

    @staticmethod
    def test_strDetect_array() -> None:
        x = np.array(["why", "video", "cross", "extra", "deal", "authority"])
        np.testing.assert_array_equal(
            stp.str_detect(x, "[aeiou]", negate=True),
            np.array([True, False, False, False, False, False]),
        )

    @staticmethod
    def test_strDetect_series() -> None:
        x = pd.Series(["why", "video", "cross", "extra", "deal", "authority"])
        pd.testing.assert_series_equal(
            stp.str_detect(x, "[aeiou]"),
            pd.Series([False, True, True, True, True, True]),
        )


class StrCount:
    # String Count
    @staticmethod
    def test_strCount_string() -> None:
        string = "video"
        assert stp.str_count(string, "[aeiou]") == 3

    @staticmethod
    def test_strCount_list() -> None:
        fruit = ["apple", "banana", "pear", "pineapple"]
        assert stp.str_count(fruit, "a") == [1, 3, 1, 1]

    @staticmethod
    def test_strCount_array() -> None:
        fruit = np.array(["apple", "banana", "pear", "pineapple"])
        np.testing.assert_array_equal(stp.str_count(fruit, "e"), np.array([1, 0, 1, 2]))

    @staticmethod
    def test_strCount_series() -> None:
        fruit = pd.Series(["apple", "banana", "pear", "pineapple"])
        pd.testing.assert_series_equal(stp.str_count(fruit, ["a", "b", "p", "p"]), pd.Series([1, 1, 1, 3]))


class StrSubset:
    # String Subset
    @staticmethod
    def test_strSubset_string() -> None:
        string = "video"
        assert stp.str_subset(string, "[aeiou]") == "video"

    @staticmethod
    def test_strSubset_list() -> None:
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        assert stp.str_subset(string, "[aeiou]") == [
            "video",
            "cross",
            "extra",
            "deal",
            "authority",
        ]

    @staticmethod
    def test_strSubset_array() -> None:
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        np.testing.assert_array_equal(stp.str_subset(string, "[aeiou]", negate=True), np.array(["why"]))

    @staticmethod
    def test_strSubset_series() -> None:
        string = pd.Series(["why", "video", "cross", "extra", "deal", "authority"])
        expected = pd.Series(["video", "cross", "authority"])
        expected.index = [1, 2, 5]
        pd.testing.assert_series_equal(stp.str_subset(string, "o"), expected)


class StrWhich:
    # String Which
    @staticmethod
    def test_strWhich_string() -> None:
        string = "video"
        assert stp.str_which(string, "[aeiou]") == 0

    @staticmethod
    def test_strWhich_list() -> None:
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        assert stp.str_which(string, "[aeiou]") == [1, 2, 3, 4, 5]

    @staticmethod
    def test_strWhich_array() -> None:
        string = ["why", "video", "cross", "extra", "deal", "authority"]
        np.testing.assert_array_equal(stp.str_which(string, "[aeiou]", negate=True), np.array([0]))

    @staticmethod
    def test_strWhich_series() -> None:
        string = pd.Series(["why", "video", "cross", "extra", "deal", "authority"])
        expected = pd.Series([1, 2, 5])
        pd.testing.assert_series_equal(stp.str_which(string, "o"), expected)


class StrReplace:
    # String Replace
    @staticmethod
    def test_strReplace_string() -> None:
        fruits = "one apple"
        assert stp.str_replace(fruits, "[aeiou]", "-") == "-ne apple"

    @staticmethod
    def test_strReplace_list() -> None:
        fruits = ["one apple", "two pears", "three bananas"]
        assert stp.str_replace(fruits, "[aeiou]", "-") == [
            "-ne apple",
            "tw- pears",
            "thr-e bananas",
        ]

    @staticmethod
    def test_strReplace_array() -> None:
        fruits = np.array(["one apple", "two pears", "three bananas"])
        np.testing.assert_array_equal(
            stp.str_replace(fruits, "b", np.nan),
            np.array(["one apple", "two pears", np.nan]),
        )

    @staticmethod
    def test_strReplace_series() -> None:
        fruits = pd.Series(["one apple", "two pears", "three bananas"])
        expected = pd.Series(["1 apple", "2 pears", "3 bananas"])
        pd.testing.assert_series_equal(stp.str_replace(fruits, ["one", "two", "three"], ["1", "2", "3"]), expected)


class StrReplaceAll:
    # String Replace All
    @staticmethod
    def test_strReplaceAll_string() -> None:
        fruits = "one apple"
        assert stp.str_replace_all(fruits, "[aeiou]", "-") == "-n- -ppl-"

    @staticmethod
    def test_strReplaceAll_list() -> None:
        fruits = ["one apple", "two pears", "three bananas"]
        assert stp.str_replace_all(fruits, "[aeiou]", "-") == [
            "-n- -ppl-",
            "tw- p--rs",
            "thr-- b-n-n-s",
        ]

    @staticmethod
    def test_strReplaceAll_array() -> None:
        fruits = np.array(["one apple", "two pears", "three bananas"])
        np.testing.assert_array_equal(
            stp.str_replace_all(fruits, "b", np.nan),
            np.array(["one apple", "two pears", np.nan]),
        )

    @staticmethod
    def test_strReplaceAll_series() -> None:
        fruits = pd.Series(["one apple", "two pears", "three bananas"])
        expected = pd.Series(["1 apple", "2 pears", "3 bananas"])
        pd.testing.assert_series_equal(
            stp.str_replace_all(fruits, ["one", "two", "three"], ["1", "2", "3"]),
            expected,
        )


class StrRemove:
    # String Remove
    @staticmethod
    def test_strRemove_string() -> None:
        fruits = "one apple"
        assert stp.str_remove(fruits, "[aeiou]") == "ne apple"

    @staticmethod
    def test_strRemove_list() -> None:
        fruits = ["one apple", "two pears", "three bananas"]
        assert stp.str_remove(fruits, "[aeiou]") == [
            "ne apple",
            "tw pears",
            "thre bananas",
        ]

    @staticmethod
    def test_strRemove_array() -> None:
        relient_k = np.array(["Devastation and Reform", "I Need You", "The Best Thing"])
        expected = np.array(["Devastation Reform", "I Need You", "The Best Thing"])
        np.testing.assert_array_equal(stp.str_remove(relient_k, "and "), expected)

    @staticmethod
    def test_strRemove_series() -> None:
        lyrics = pd.Series(["Come", "Right", "Out", "and", "Say", "It"])
        expected = pd.Series(["Come", "Rght", "Out", "and", "Say", "t"])
        pd.testing.assert_series_equal(stp.str_remove(lyrics, "[iI]"), expected)


class StrRemoveAll:
    # String Remove All
    @staticmethod
    def test_strRemoveAll_string() -> None:
        fruits = "one apple"
        assert stp.str_remove_all(fruits, "[aeiou]") == "n ppl"

    @staticmethod
    def test_strRemoveAll_list() -> None:
        fruits = ["one apple", "two pears", "three bananas"]
        assert stp.str_remove_all(fruits, "[aeiou]") == ["n ppl", "tw prs", "thr bnns"]

    @staticmethod
    def test_strRemoveAll_array() -> None:
        relient_k = np.array(["Devastation and Reform", "I Need You", "The Best Thing"])
        expected = np.array(["Devastation Reform", "I Need You", "The Best Thing"])
        np.testing.assert_array_equal(stp.str_remove_all(relient_k, "and "), expected)

    @staticmethod
    def test_strRemoveAll_series() -> None:
        lyrics = pd.Series(["Come", "Right", "Out", "and", "Say", "It"])
        expected = pd.Series(["Come", "Rght", "Out", "and", "Say", "t"])
        pd.testing.assert_series_equal(stp.str_remove_all(lyrics, "[iI]"), expected)


class TestStrSplit:
    # String Split
    @staticmethod
    def test_strSplit_string() -> None:
        string = "guinea pig and farts"
        assert stp.str_split(string, " and ") == ["guinea pig", "farts"]

    @staticmethod
    def test_strSplit_list() -> None:
        fruits = [
            "apples and oranges and pears and bananas",
            "pineapples and mangos and guavas",
        ]
        expected = [
            ["apples", "oranges", "pears", "bananas"],
            ["pineapples", "mangos", "guavas", ""],
        ]
        assert stp.str_split(fruits, " and ", simplify=True) == expected

    @staticmethod
    def test_strSplit_array() -> None:
        fruits = np.array(
            [
                "apples and oranges and pears and bananas",
                "pineapples and mangos and guavas",
            ]
        )
        expected = np.array(
            [
                ["apples", "oranges and pears and bananas"],
                ["pineapples", "mangos and guavas"],
            ]
        )
        assert all(stp.str_split(fruits, " and ", n=1)[0] == expected[0])
        assert all(stp.str_split(fruits, " and ", n=1)[1] == expected[1])

    @staticmethod
    def test_strSplit_series() -> None:
        fruits = pd.Series(
            [
                "apples and oranges and pears and bananas",
                "pineapples and mangos and guavas",
            ]
        )
        actual = stp.str_split(fruits, " and ", n=5)
        expected = pd.Series(
            [
                ["apples", "oranges", "pears", "bananas"],
                ["pineapples", "mangos", "guavas"],
            ]
        )
        pd.testing.assert_series_equal(actual, expected)


class TestStrSplitFixed:
    # String Split Fixed
    @staticmethod
    def test_strSplitFixed_string() -> None:
        string = "guinea pig and farts"
        assert stp.str_split_fixed(string, " and ") == ["guinea pig", "farts"]

    @staticmethod
    def test_strSplitFixed_list() -> None:
        fruits = [
            "apples and oranges and pears and bananas",
            "pineapples and mangos and guavas",
        ]
        expected = [
            ["apples", "oranges", "pears and bananas"],
            ["pineapples", "mangos", "guavas"],
        ]
        assert stp.str_split_fixed(fruits, " and ", 2) == expected

    @staticmethod
    def test_strSplitFixed_array() -> None:
        fruits = np.array(
            [
                "apples and oranges and pears and bananas",
                "pineapples and mangos and guavas",
            ]
        )
        expected = np.array(
            [
                ["apples", "oranges and pears and bananas"],
                ["pineapples", "mangos and guavas"],
            ]
        )
        np.testing.assert_array_equal(stp.str_split_fixed(fruits, " and ", n=1), expected)

    @staticmethod
    def test_strSplitFixed_series() -> None:
        fruits = pd.Series(
            [
                "apples and oranges and pears and bananas",
                "pineapples and mangos and guavas",
            ]
        )
        expected = pd.DataFrame(
            [
                ["apples", "oranges", "pears", "bananas"],
                ["pineapples", "mangos", "guavas", ""],
            ]
        )
        pd.testing.assert_frame_equal(stp.str_split_fixed(fruits, " and ", 3), expected)


class TestStrSplitN:
    # String Split N
    @staticmethod
    def test_strSplitN_string() -> None:
        string = "guinea pigs and farts"
        assert stp.str_split_n(string, " ", 1) == "pigs"

    @staticmethod
    def test_strSplitN_list() -> None:
        fruits = [
            "apples and oranges and pears and bananas",
            "pineapples and mangos and guavas",
        ]
        assert stp.str_split_n(fruits, " and ", 0) == ["apples", "pineapples"]

    @staticmethod
    def test_strSplitN_array() -> None:
        fruits = np.array(
            [
                "apples and oranges and pears and bananas",
                "pineapples and mangos and guavas",
            ]
        )
        assert stp.str_split_n(fruits, " and ", 3)[0] == "bananas"
        assert stp.str_split_n(fruits, " and ", 3)[1] is np.nan

    @staticmethod
    def test_strSplitN_series() -> None:
        fruits = pd.Series(
            [
                "apples and oranges and pears and bananas",
                "pineapples and mangos and guavas",
            ]
        )
        pd.testing.assert_series_equal(stp.str_split_n(fruits, " and ", 3), pd.Series(["bananas", np.nan], name=3))


class TestStrStarts:
    # String Starts
    @staticmethod
    def test_strStarts_string() -> None:
        string = "guinea pigs and farts"
        assert stp.str_starts(string, "guinea") == True

    @staticmethod
    def test_strStarts_list() -> None:
        fruit = ["apple", "banana", "pear", "pineapple"]
        assert stp.str_starts(fruit, "p") == [False, False, True, True]

    @staticmethod
    def test_strStarts_array() -> None:
        fruit = np.array(["apple", "banana", "pear", "pineapple"])
        np.testing.assert_array_equal(
            stp.str_starts(fruit, "p", negate=True),
            np.array([True, True, False, False]),
        )

    @staticmethod
    def test_strStarts_series() -> None:
        fruit = pd.Series(["apple", "banana", "pear", "pineapple"])
        pd.testing.assert_series_equal(stp.str_starts(fruit, "[aeiou]"), pd.Series([True, False, False, False]))


class TestStrEnds:
    # String Ends
    @staticmethod
    def test_strEnds_string() -> None:
        string = "guinea pigs and farts"
        assert stp.str_ends(string, "farts") is True

    @staticmethod
    def test_strEnds_list() -> None:
        fruit = ["apple", "banana", "pear", "pineapple"]
        assert stp.str_ends(fruit, "e") == [True, False, False, True]

    @staticmethod
    def test_strEnds_array() -> None:
        fruit = np.array(["apple", "banana", "pear", "pineapple"])
        np.testing.assert_array_equal(stp.str_ends(fruit, "e", True), np.array([False, True, True, False]))

    @staticmethod
    def test_strEnds_series() -> None:
        fruit = pd.Series(["apple", "banana", "pear", "pineapple"])
        pd.testing.assert_series_equal(stp.str_ends(fruit, "[aeiou]"), pd.Series([True, True, False, True]))


class TestStrExtract:
    # String Extract
    @staticmethod
    def test_strExtract_string() -> None:
        number = "219 733 8965"
        assert stp.str_extract(number, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == number

    @staticmethod
    def test_strExtract_list() -> None:
        shopping_list = ["apples x4", "bag of flour", "bag of sugar", "milk x2"]
        assert stp.str_extract(shopping_list, "\\d") == ["4", None, None, "2"]

    @staticmethod
    def test_strExtract_array() -> None:
        shopping_list = np.array(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        np.testing.assert_array_equal(
            stp.str_extract(shopping_list, "[a-z]{1,4}"),
            np.array(["appl", "bag", "bag", "milk"]),
        )

    @staticmethod
    def test_strExtract_series() -> None:
        shopping_list = pd.Series(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        pd.testing.assert_series_equal(
            stp.str_extract(shopping_list, "[a-z]+"),
            pd.Series(["apples", "bag", "bag", "milk"]),
        )


class TestStrExtractAll:
    # String Extract All
    @staticmethod
    def test_strExtractAll_string() -> None:
        number = "219 733 8965"
        assert stp.str_extract_all(number, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == [("219", "733", "8965")]

    @staticmethod
    def test_strExtractAll_list() -> None:
        shopping_list = ["apples x4", "bag of flour", "bag of sugar", "milk x2"]
        assert stp.str_extract_all(shopping_list, "\\b[a-z]+\\b") == [
            ["apples"],
            ["bag", "of", "flour"],
            ["bag", "of", "sugar"],
            ["milk"],
        ]

    @staticmethod
    def test_strExtractAll_array() -> None:
        shopping_list = np.array(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        np.testing.assert_array_equal(
            stp.str_extract_all(shopping_list, "\\b[a-z]+\\b", simplify=True),
            np.array(
                [
                    ["apples", "", ""],
                    ["bag", "of", "flour"],
                    ["bag", "of", "sugar"],
                    ["milk", "", ""],
                ]
            ),
        )

    @staticmethod
    def test_strExtractAll_series() -> None:
        shopping_list = pd.Series(["apples x4", "bag of flour", "bag of sugar", "milk x2"])
        pd.testing.assert_series_equal(
            stp.str_extract_all(shopping_list, "\\d", simplify=True),
            pd.Series(["4", "", "", "2"], name="match"),
        )


class TestStrMatch:
    # String Match
    @staticmethod
    def test_strMatch_string() -> None:
        string = "<a> <b>"
        assert stp.str_match(string, pattern="<(.*?)> <(.*?)>") == ["<a> <b>", "a", "b"]

    @staticmethod
    def test_strMatch_list() -> None:
        strings = [
            " 219 733 8965",
            "329-293-8753 ",
            "banana",
            "595 794 7569",
            "387 287 6718",
            "apple",
            "233.398.9187  ",
            "482 952 3315",
            "239 923 8115 and 842 566 4692",
            "Work: 579-499-7527",
            "$1000",
            "Home: 543.355.3679",
        ]
        expected = [
            ["219 733 8965", "219", "733", "8965"],
            ["329-293-8753", "329", "293", "8753"],
            [None, None, None, None],
            ["595 794 7569", "595", "794", "7569"],
            ["387 287 6718", "387", "287", "6718"],
            [None, None, None, None],
            ["233.398.9187", "233", "398", "9187"],
            ["482 952 3315", "482", "952", "3315"],
            ["239 923 8115", "239", "923", "8115"],
            ["579-499-7527", "579", "499", "7527"],
            [None, None, None, None],
            ["543.355.3679", "543", "355", "3679"],
        ]
        assert stp.str_match(strings, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == expected

    @staticmethod
    def test_strMatch_array() -> None:
        string = np.array(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = np.array(
            [
                ["<a> <b>", "a", "b"],
                ["<a> <>", "a", ""],
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ]
        )
        np.testing.assert_array_equal(stp.str_match(string, "<(.*?)> <(.*?)>"), expected)

    @staticmethod
    def test_strMatch_series() -> None:
        string = pd.Series(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = pd.DataFrame(
            {
                "whole_match": ["<a> <b>", "<a> <>", None, None, None],
                "match": ["a", "a", np.nan, np.nan, np.nan],
            }
        )
        pd.testing.assert_frame_equal(stp.str_match(string, "<(.*?)> <(.*?)>"), expected)


class TestStrMatchAll:
    # String Match All
    @staticmethod
    def test_strMatchAll_string() -> None:
        string = "239 923 8115 and 842 566 4692"
        expected = [
            ["239 923 8115", "239", "923", "8115"],
            ["842 566 4692", "842", "566", "4692"],
        ]
        assert stp.str_match_all(string, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == expected

    @staticmethod
    def test_strMatchAll_list() -> None:
        strings = [
            " 219 733 8965",
            "329-293-8753 ",
            "banana",
            "595 794 7569",
            "387 287 6718",
            "apple",
            "233.398.9187  ",
            "482 952 3315",
            "239 923 8115 and 842 566 4692",
            "Work: 579-499-7527",
            "$1000",
            "Home: 543.355.3679",
        ]
        expected = [
            ["219 733 8965", "219", "733", "8965"],
            ["329-293-8753", "329", "293", "8753"],
            ["", "", "", ""],
            ["595 794 7569", "595", "794", "7569"],
            ["387 287 6718", "387", "287", "6718"],
            ["", "", "", ""],
            ["233.398.9187", "233", "398", "9187"],
            ["482 952 3315", "482", "952", "3315"],
            ["239 923 8115", "239", "923", "8115"],
            ["842 566 4692", "842", "566", "4692"],
            ["579-499-7527", "579", "499", "7527"],
            ["", "", "", ""],
            ["543.355.3679", "543", "355", "3679"],
        ]
        assert stp.str_match_all(strings, "([2-9][0-9]{2})[- .]([0-9]{3})[- .]([0-9]{4})") == expected

    @staticmethod
    def test_strMatchAll_array() -> None:
        x = np.array(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = np.array(
            [
                ["<a>", "a"],
                ["<b>", "b"],
                ["<a>", "a"],
                ["", ""],
                ["<a>", "a"],
                ["", ""],
                ["", ""],
            ]
        )
        np.testing.assert_array_equal(stp.str_match_all(x, "<(.*?)>"), expected)

    @staticmethod
    def test_strMatchAll_series() -> None:
        x = pd.Series(["<a> <b>", "<a> <>", "<a>", "", None])
        expected = pd.DataFrame(
            [
                ["<a>", "a"],
                ["<b>", "b"],
                ["<a>", "a"],
                ["<>", ""],
                ["<a>", "a"],
                ["", ""],
                ["", ""],
            ]
        )
        expected.columns = ["whole_match", 0]
        pd.testing.assert_frame_equal(stp.str_match_all(x, "<(.*?)>"), expected)


# String Equal
class TestStrEqual:
    @staticmethod
    def test_strEqual_str() -> None:
        string1 = "\u03a9"
        string2 = "\u2126"
        assert stp.str_equal(string1, string2) is True
        assert stp.str_equal(string1, string2, True) is True

    @staticmethod
    def test_strEqual_list() -> None:
        string1 = ["\u00e1"]
        string2 = ["a\u0301"]
        np.testing.assert_array_equal(stp.str_equal(string1, string2, True), [True])
        np.testing.assert_array_equal(stp.str_equal(string1, string2), [True])

    @staticmethod
    def test_strEqual_array() -> None:
        string1 = np.array(["guinea", "pigs", "and", "farts"])
        string2 = np.array(["Guinea", "pigs", "&", "farts"])
        np.testing.assert_array_equal(stp.str_equal(string1, string2), np.array([False, True, False, True]))
        np.testing.assert_array_equal(stp.str_equal(string1, string2, True), np.array([True, True, False, True]))

    @staticmethod
    def test_strEqual_series() -> None:
        string1 = pd.Series(["Farts", "and", "guinea", "pigs"])
        string2 = pd.Series(["farts", "&", "guinea", "pigs"])
        pd.testing.assert_series_equal(pd.Series([True, False, True, True]), stp.str_equal(string1, string2, True))
        pd.testing.assert_series_equal(pd.Series([False, False, True, True]), stp.str_equal(string1, string2))


class TestStrUnique:
    @staticmethod
    def test_strUnique_string() -> None:
        string = "guinea pig"
        assert stp.str_unique(string) == "guinea p"
        assert stp.str_unique(string, True) == "guinea p"

    @staticmethod
    def test_strUnique_list() -> None:
        string = ["a", "b", "c", "b", "a"]
        assert stp.str_unique(string, 1) == ["a", "b", "c"]
        assert stp.str_unique(string, 4) == ["a", "b", "c"]

    @staticmethod
    def test_strUnique_array() -> None:
        string = np.array(["motley", "mötley", "pinguino", "pingüino"])
        np.testing.assert_array_equal(stp.str_unique(string, 1), np.array(["motley", "pinguino"]))
        np.testing.assert_array_equal(stp.str_unique(string, 4), string)

    @staticmethod
    def test_strUnique_series() -> None:
        string = pd.Series(["What", "what", "is", "love"])
        pd.testing.assert_series_equal(
            stp.str_unique(string, 1),
            pd.Series(["What", "is", "love"], index=[0, 1, 2]),
        )
        pd.testing.assert_series_equal(stp.str_unique(string, 4), string)


if __name__ == "__main__":
    pytest.main()
