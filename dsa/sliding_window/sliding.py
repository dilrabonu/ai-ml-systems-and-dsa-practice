def length_og_logest_substrings(s):
    last_pos = {}
    left = 0
    best = 0

    for right, ch in enumerate(s):
        if ch in last_pos and last_pos[ch] >= left:
            left = last_pos[ch] + 1
        last_pos[ch] = right
        best = max(best, right- left + 1)

    return best

def max_sunm_subarray(nums, k):
    if k > len(nums):
        return None

    window_sum = sum(nums[:k])
    best = window_sum

    for right in range(k, len(nums)):
        window_sum += nums[right]
        window_sum -= nums[right - k]
        best = max(best, window_sum)

    return best


