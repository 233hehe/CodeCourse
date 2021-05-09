class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        def check(mid):
            i = cnt = 0
            while i < n and cnt < m:
                cur = 1 if bloomDay[i] <= mid else 0
                j = i
                if cur > 0:
                    while cur < k and j + 1 < n and bloomDay[j+1] <= mid:
                        j += 1
                        cur += 1
                    if cur == k:
                        cnt += 1
                    i = j + 1
                else:
                    i += 1
            return cnt >= m

        n = len(bloomDay)
        if n < m * k:
            return -1
        lower, upper = m * k, max(bloomDay)
        while lower < upper:
            mid = lower + upper >> 1
            if check(mid):
                upper = mid
            else:
                lower = mid + 1
        return upper
