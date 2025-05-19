def is_prime(n, a=None, b=None):
    def has_divisor_in_range(num, start, end, step):
        for i in range(start, end, step):
            if num % i == 0:
                return True
        return False

    n = int(n)
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    return not has_divisor_in_range(n, 5, int(n**0.5) + 1, 6)