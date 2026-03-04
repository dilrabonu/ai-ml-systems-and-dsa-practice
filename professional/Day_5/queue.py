from collection import deque

queue = deque()
queue.append("A")
queue.append("B")
queue.append("C")

# take the first element
item = queue.popleft()

# to see the first lement with peek
front = queue[0]

# check is empty len(queue) == 0

if not queue:
    print("Queue is empty")