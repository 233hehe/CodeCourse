"""[summary]
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isidentical(self, node_a, node_b):
        if not node_a and not node_b:
            return True
        if node_a is None or node_b is None:
            return False
        return (node_a.val == node_b.val) and (self.isidentical(node_a.left, node_b.left)) and (self.isidentical(node_a.right, node_b.right))
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if not s:
            return False
        if self.isidentical(s, t):
            return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)