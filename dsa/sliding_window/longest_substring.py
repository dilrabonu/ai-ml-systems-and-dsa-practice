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

