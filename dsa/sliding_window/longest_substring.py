def longest_unique_substring(s: str) -> int:
    chars: set[str] = set()
    left = 0
    best = 0

    for right in range(len(s)):
        while s[right] in chars:
            chars.remove(s[left])
            left += 1

        chars.add(s[right])
        best = max(best, right - left + 1)

    return best

# Approach B
def longest_unique_substring_fast(s):
    last_seen: dict[str, int] = {}
    left = 0
    best = 0

    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best