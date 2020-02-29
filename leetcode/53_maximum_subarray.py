"""
Given an integer array nums, find the contiguous subarray (containing at least one number)
 which has the largest sum and return its sum.

Example:

Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

"""


class Solution:
    def maxSubArray(self, nums) -> int:
        max_len: int = len(nums)
        max_res: int = nums[0]
        max_now: int = 0
        i: int = 0
        while i < max_len:
            max_now += nums[i]
            if max_now >= max_res:
                max_res = max_now
            if max_now < 0:
                max_now = 0
            i += 1
        return max_res
