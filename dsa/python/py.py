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
students = [("Akmal", 45), ("Dilya", 92), ("Benny", 78)]
sorted_students = sorted(students, key=lambda student: student[1])
print(sorted_students)

# enumerate
fruits = ["apple", "cherry", "pear", "peach"]
i = 0
for fruit in fruits:
    print(i, fruit)
    i += 1

for i, fruit in enumerate(fruits):
    print(i, fruit)

# zip
names = ["Abbos", "Ali", "Vali"]
ages = [31, 41, 51]

for name, age in zip(names, ages):
    print(f"{name} - {age} years old") 

class Car:
    def __init__(self, colour, brend, cost):
        self.colour = colour
        self.brend = brend
        self.cost = cost

    def run(self):
        print(f"{self.colour} {self.brend} is running")

    def info(self):
        print(f"{self.colour} {self.brend} is {self.cost} so'm")

car1 = Car("Black", "BMW", 50000)
car1.info()

# decorator - wrappers and add behavior without changing the code

import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, *kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

@timer
def train_epoch():
    time.sleep(1)
    return "done"

train_epoch()
