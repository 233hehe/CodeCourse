"""
Determine whether an integer is a palindrome.
 An integer is a palindrome when it reads the same backward as forward.
"""


class Solution:
    def isPalindrome(self, x: int) -> bool:
        x_str = str(x)
        x_reverse = x_str[::-1]
        return x_str == x_reverse
