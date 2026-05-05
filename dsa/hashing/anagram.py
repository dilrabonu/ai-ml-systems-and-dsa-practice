# task: given two strings s and t, return true if t is an anagram of s, and false otherwise.
from collections import defaultdict
def groupAnagrams(strs):
    d = defaultdict(list)

    for word in strs:
        key = "".join(sorted(word))
        d[key].append(word)

    return list(d.values())

# without default dict
def groupAnagrams(strs):
    d = {}
    
    for word in strs:
        key = "".join(sorted(word))
        if key not in d:
            d[key] = []
        d[key].append(word)
    
    return list(d.values())

def groupAnagram(strs):
    d = defaultdict(list)
    for word in strs:
        key = "".join(sorted(word))
        d[key].append(word)

    return list(d.values())