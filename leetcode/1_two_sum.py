"""Given an array of integers, return indices of the two numbers
such that they add up to a specific target.

You may assume that each input would have exactly one solution,
and you may not use the same element twice.
"""
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashdict = dict()
        for i in range(len(nums)):
            complement: int = target - nums[i]
            if hashdict.get(complement) is not None:
                return [hashdict.get(complement), i]
            hashdict.update({nums[i]: i})

