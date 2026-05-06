import pytest
from dsa.sliding_window.max_subarray import max_sum_subarray, max_sum_brute
from dsa.sliding_window.longest_substring import longest_unique_substring
from dsa.sliding_window.min_window_substring import min_window

class TestMaxSumSubarray:
    def test_basic(self):
        assert max_sum_subarray([2, 1, 5, 1, 3, 2], 3) == 9

    def test_empty(self):
        assert max_sum_subarray([], 3) == 0

    def test_k_larger_than_array(self):
        assert max_sum_subarray([1, 2], 5) == 0

    def test_all_negatives(self):
        assert max_sum_subarray([-1, -2, -3, -4], 2) == -3

    def test_matches_brute_force_random(self):
        import random
        random.seed(7)
        for _ in range(50):
            n = random.randint(1, 50)
            k = random.randint(1, n)
            arr = [random.randint(-100, 100) for _ in range(n)]
            assert max_sum_subarray(arr, k) == max_sum_brute(arr, k)

@pytest.mark.parametrize("s,expected", [
    ("abcabcbb", 3),
    ("bbbbb",   1),
    ("pwwkew",  3),
    ("",        0),
    (" ",       1),
    ("dvdf",    3),                        # tricky case
])
def test_longest_unique(s, expected):
    assert longest_unique_substring(s) == expected

def test_min_window_basic():
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"

def test_min_window_no_solution():
    assert min_window("a", "aa") == ""