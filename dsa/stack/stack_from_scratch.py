from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._data: list[T] = []

    def push(self, x: T) -> None:
        self._data.append(x)

    def pop(self) -> T:
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> T:
        if not self._data:
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self) -> bool:
        return not self._data

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Stack{self._data!r}"

# Linked List Implementation
class _Node:
    __slots__ = ("val", "next")
    def __init__(self, val, nxt=None):
        self.val = val
        self.nxt = nxt
class LinkedStack(Generic[T]):
    def __init__(self) -> None:
        self._top: Optional[_Node] = None
        self._size : int = 0

    def push(self, x: T) -> None:
        self._top = _node(x, self._top)
        self._size += 1

    def pop(self) -> T:
        if self._top is None:
            raise IndexError("pop from empty stack")
        val = self._top.val
        self._top = self._top.next
        self._size -= 1
        return val

    def peek(self) -> T:
        if self._top is None:
            raise IndexError("peek from empty stack")
        return self._top.val

    def __len__(self) -> int:
        return self._size
