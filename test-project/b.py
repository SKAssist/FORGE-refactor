from is_prime import is_prime

nums = list(range(1, 20))
prime_nums = []

for n in nums:
    if is_prime(n):
        prime_nums.append(n)

print("Prime numbers between 1 and 20:", prime_nums)