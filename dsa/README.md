# Data Structures & Algorithms (DSA)

This module contains my structured practice of core data structures and algorithms, focusing on problem-solving, efficiency, and interview readiness.

## Topics Covered
- Arrays & Strings
- Hashing
- Sliding Window
- Recursion & Backtracking
- Graphs & Trees
- Dynamic Programming

## Goals
- Build strong algorithmic thinking
- Improve time and space complexity optimization
- Prepare for technical interviews (FAANG-level)

## Structure
Each subfolder contains:
- Problem implementations
- Clean, readable solutions
- Optimized approaches

## Example
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i