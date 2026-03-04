"""
 pseudo code
def isValid(s):
    HashMap create 
    empty stack create
    iterate for every char in s:
        if char is openeing push tostack
        else:
            if stack closed barcket -> false
            stack pop() != suitable opening barcket -> false
        stack empty True
        stack is not empty False
"""
def isValid(s):
    mapping = { ")": "(", "]": "[", "}": "{" }
    stack = []
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if top != mapping[char]:
                return False
        else:
            stack.append(char)
    return len(stack) == 0

# Generate Paranthesis
def generateParanthesis(n: int) -> list[str]:
    result = []

    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return

        if open_count < n:
            backtrack(current + ')', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count +1)
    backtrack('', 0, 0)
    return result
    

