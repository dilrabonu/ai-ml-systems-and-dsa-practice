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