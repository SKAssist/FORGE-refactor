def is_prime(n):
    """
    Determine if a number is prime. A prime number is greater than 1 and
    has no divisors other than 1 and itself.

    Args:
        n: An integer to check for primality

    Returns:
        bool: True if n is prime, False otherwise
    """
    if not is_greater_than_one(n):
        return False

    if n in [2, 3]:
        return True

    if is_divisible_by_2_or_3(n):
        return False

    return not has_divisor_in_range(n)

def is_greater_than_one(n):
    return n > 1

def is_divisible_by_2_or_3(n):
    return n % 2 == 0 or n % 3 == 0

def has_divisor_in_range(n):
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return True
        i += 6
    return False