# Simple dictionary
def build_freq(arr):
    freq = {}
    for num in arr:
        if num in freq:
            freq[num] += 1
        else:
            freq[num] = 1
    return freq

# Dict with get()
def build_freq_map(arr):
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) +1
    return freq

# Defaultdict
from collections import defaultdict

def build_freq(arr):
    freq = defaultdict(int)
    for num in arr:
        freq[num] += 1
    return dict(freq)

# Counter
from collections import Counter
arr = [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1]
freq = Counter(arr)

print(freq)

# find uniq char
def UniqChar(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    return -1

def TwoSum(nums, target):
    seen = {}
    for i , num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]

        seen[num] = i
    return []    

def TwoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []

def UniqChar(s):
    freq = {}
    for char in s:
        ferq[char] = freq.get(char, 0) + 1

    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    return -1

def TwoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def twosum_sorted(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Anagram
def isAnagram(s, t):
    if len(s) != len(t):
        return False

    freq_s = {}
    freq_t = {}

    for char in s:
        freq_s[char] = freq_s.get(char, 0) + 1
        
    for char in t:
        freq_t[char] = freq_t.get(char, 0) + 1

    return freq_s == freq_t

def groupsAnagram(strs):
    groups = defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))
        groups[key].append(s)

    return list(groups.values())

def containDuplicate(nums):
    seen = set()
    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False


