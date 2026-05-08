class MinStack:
    def __init__(self) -> None:
        self._stack: list[int] = []
        self._mins: list[int] = []

    def push(self, x: int) -> None:
        self._stack.append(x)

        new_min = x if not self._mins else min(x, self._mins[-1])
        self._mins.append(new_min)

    def pop(self) -> None:
        self._stack.pop()
        self._mins.pop()

    def top(self) -> int:
        return sel._stack[-1]
    
    def get_min(self) -> int:
        return self._mins[-1]