from collection import Counter

def min_window(s, t):
    if not t or not s:
        return ""

    need = Counter(t)
    missing = len(t)
    left = 0
    best_lo, best_hi = 0, float("inf")

    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1

        while missing == 0:
            if right - left < best_hi - best_lo:
                best_lo, best_hi = left , right
            left_ch = s[left]
            need[left_ch] += 1
            if need[left_ch] > 0:
                missing += 1
            left += 1
    
    return "" if best_hi == float("inf") else s[best_lo: best_hi + 1]
    
    