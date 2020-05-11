"""
Given a 32-bit signed integer, reverse digits of an integer.
"""

max_int = 2 ** 31 - 1
min_int = -(2 ** 31)
upper_bound = int(max_int / 10)
lower_bound = int(min_int / 10)


def reverse(x: int) -> int:
    if x == 0:
        return 0
    ret = int(("-" if x < 0 else "") + str(abs(x))[::-1].lstrip("0"))
    if abs(ret) > (2 ** 31 - 1):
        return 0
    return ret
