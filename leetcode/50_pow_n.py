"""Implement pow(x, n), which calculates x raised to the power n (xn).

"""
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0: 
            return 1
        elif n == 1:
            return x
        elif n < 0:
            n = -n
            half = self.myPow(x, n // 2)
            if n % 2 == 0:
                return 1 / (half * half)
            else:
                return 1 / (half * half * x)
        else:
            half = self.myPow(x, n // 2)
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x