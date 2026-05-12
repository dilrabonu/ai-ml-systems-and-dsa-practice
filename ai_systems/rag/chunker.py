"""
Three chunking strategies, from naive to smart.

Trade-offs:
  word-window      : simplest, breaks mid-sentence (bad)
  sentence-window  : respects sentence boundaries (default)
  semantic         : splits where meaning shifts (best, slowest)
"""
