from is_prime import is_prime

grid = [
    [2, 4, 5],
    [6, 7, 9],
    [10, 11, 13]
]

prime_count = 0

for row in grid:
for num in row:
    if is_prime(num):
        prime_count += 1

    print("Number of primes in the grid:", prime_count)