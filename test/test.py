import torch
from signax import _impl

print(dir(_impl.cpuSignatureForward()))
# from typing import List
#
#
# class Solution:
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         n = len(nums)
#         index_map = dict(zip(nums, range(n)))
#         res = []
#
#         for i in range(n):
#             for j in range(i, n):
#                 target = - (i + j)
#                 if target in index_map and index_map[target] > j:
#                     res.append([nums[i], nums[j], nums[target]])
#
#         return res