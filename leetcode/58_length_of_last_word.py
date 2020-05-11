"""
Given a string s consists of upper/lower-case alphabets and empty space characters ' ',
return the length of last word
(last word means the last appearing word if we loop from left to right) in the string.

If the last word does not exist, return 0.

Note: A word is defined as a maximal substring consisting of non-space characters only.

"""


class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        split_list = s.split(" ")
        i = len(split_list) - 1
        all_flag = True
        while i >= 0:
            if split_list[i] == "":
                i = i - 1
            else:
                all_flag = False
                return len(split_list[i])
                break
        if all_flag:
            return 0
