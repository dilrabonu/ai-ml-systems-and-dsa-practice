fruits = ["apple", "banana", "strawberry"]
comp = [f.upper() for f in fruits]
print(comp)

even = [n for n in range(1, 11) if n % 2 == 0]
print(even)

toq = [n for n in range(1, 11) if n  % 2 != 0]
print(toq)

even = [n for n in range(2,22) if n % 2 == 0]
print(even)

# lambda:
square = lambda x: x ** 2
print(square(5))