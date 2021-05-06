class Solution:
    def decode(self, encoded: List[int], first: int) -> List[int]:
        output_list = [first]
        left = first
        for encode_int in encoded:
            right = encode_int ^ left
            left = right
            output_list.append(right)
        return output_list
